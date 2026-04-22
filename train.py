from dataset import create_dataframe, CTLesionSegmentation, split_dataset, find_mean_std, transforms
from torch.utils.data import DataLoader
from models import UNet
from config import *
import torch
from torch import nn
import torch.nn.functional as F
import wandb



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_loss(pred, target, smooth=1):

    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()

def dice_score(pred, target, smooth=1):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()  # binarise
    
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice.mean()

def loss_function(pred, target):

    if LOSS_FUNCTION == 'BCE':

        return nn.BCEWithLogitsLoss()(pred, target)
    
    elif LOSS_FUNCTION == 'DICE':

        return dice_loss(pred, target)
    
    elif LOSS_FUNCTION == 'BOTH':

        return nn.BCEWithLogitsLoss()(pred, target) + dice_loss(pred, target)


def train(epochs, model, train_dataloader, val_dataloader, optimizer, name, run):

    best_val_dice = 0.
    patience_counter = 0 #used for early stopping
    min_delta = 0.005 # Minimum change in loss to qualify as an improvement

    for epoch in range(epochs):
        model.train()
        running_loss = 0.

        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() #zero the grads in each epoch
            output = model(inputs)

            #compute the loss
            loss = loss_function(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.
        val_dice = 0.

        with torch.no_grad():  
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                output = model(inputs)
                loss   = loss_function(output, labels)
                val_loss += loss.item()
                val_dice += dice_score(output, labels)

        val_dice_mean = val_dice / len(val_dataloader)

        if val_dice_mean > best_val_dice + min_delta:
            patience_counter = 0
            best_val_dice = val_dice_mean
            torch.save(model.state_dict(), f'checkpoints/best_model_{name}.pth')
        else:
            patience_counter += 1       
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_dataloader):.4f}, Val Loss: {val_loss/len(val_dataloader):.4f},  Val Dice: {val_dice_mean:.4f}')

        run.log({"epoch": epoch + 1, "train_loss": running_loss / len(train_dataloader), "val_loss": val_loss / len(val_dataloader),  "val_dice": val_dice_mean})
   
    run.summary["best_val_dice"] = best_val_dice
    run.summary["stopped_at_epoch"] = epoch + 1

    


if __name__ == "__main__":

    #---PROCESS---
    #Step 1: create dataframe based on the image paths
    df = create_dataframe()

    #Step 2: Split the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)

    #Step 3: Create Dataset objects
    mean_d, std_d = find_mean_std()
    train_transform, val_test_transform = transforms(mean_d, std_d)
    train_ds = CTLesionSegmentation(X_train, y_train, transform=train_transform)
    val_ds   = CTLesionSegmentation(X_val,   y_val,   transform=val_test_transform)

    #Step 4:  Call DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    for depth_values in SWIN_DEPTH:

        run = wandb.init(
        entity="nasosk16-city-university-of-london",
        name=f"swindepth_{depth_values}",
        project="CT-scan-segmentation",

        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "embed_dim": EMBED_DIM,
            "win": WIN,
            "heads": HEADS,
            "swin_depth": depth_values,
            'patch_size': PATCH_SIZE,
            'LOSS_FUNCTION': LOSS_FUNCTION,
            'optimizer': OPTIMIZER,
            'dropout': DROPOUT
        },
        )

        model = UNet(256, PATCH_SIZE, 1, EMBED_DIM, WIN, HEADS, depth_values, 1, DROPOUT).to(device)

        if OPTIMIZER == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        elif OPTIMIZER == 'ADAMW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
        elif OPTIMIZER == 'SGD_M':
            optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

        train(EPOCHS, model, train_dataloader, val_dataloader, optimizer, run.name, run)
        run.finish()



    

