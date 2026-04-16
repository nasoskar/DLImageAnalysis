from dataset import create_dataframe, CTLesionSegmentation, split_dataset, train_transform, val_test_transform
from torch.utils.data import DataLoader
from models import UNet
import torch
from torch import nn
import torch.nn.functional as F
import wandb

########### PARAMETERS ############
BATCH_SIZE = 16
epochs = 10
LR = 0.001
# WEIGHT_DECAY = 1e-4 (Depends on your optmiser)
NUM_CLASSES = 1
patch_size = 4
win = 8
heads = 8
swin_depth  = 2
embed_dim   = 96


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()

def loss_function(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)  # handles sigmoid internally
    d_loss = dice_loss(pred, target)
    
    return bce + d_loss


def train(epochs, model, train_dataloader, val_dataloader, optimizer):

    best_val_loss = float('inf')
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

        with torch.no_grad():  
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                output = model(inputs)
                loss   = loss_function(output, labels)
                val_loss += loss.item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_dataloader):.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}')

        run.log({"epoch": epoch + 1, "train_loss": running_loss / len(train_dataloader), "val_loss": val_loss / len(val_dataloader)})

    


if __name__ == "__main__":

    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="nasosk16-city-university-of-london",
    # Set the wandb project where this run will be logged.
    project="CT-scan-segmentation",
    # Track hyperparameters and run metadata.
    config={
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "embed_dim": embed_dim,
        "win": win,
        "heads": heads,
        "swin_depth": swin_depth
    },
    )

    #---PROCESS---
    #Step 1: create dataframe based on the image paths
    df = create_dataframe()

    #Step 2: Split the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)

    #Step 3: Create Dataset objects
    train_ds = CTLesionSegmentation(X_train, y_train, transform=train_transform)
    val_ds   = CTLesionSegmentation(X_val,   y_val,   transform=val_test_transform)
    test_ds  = CTLesionSegmentation(X_test,  y_test,  transform=val_test_transform)

    #Step 4:  Call DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    model = UNet(256, patch_size, 1, embed_dim, win, heads, swin_depth, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(epochs, model, train_dataloader, val_dataloader, optimizer)
    run.finish()



    

