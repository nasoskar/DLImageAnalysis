from train import UNet, loss_function, dice_loss, create_dataframe, split_dataset
from dataset import CTLesionSegmentation, val_test_transform
from config import *
from torch.utils.data import DataLoader
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()  # binarise
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * intersection + 1) / (union + 1)
    return dice.mean().item()

def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()  # binarise
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
    iou = (intersection + 1) / (union + 1)
    return iou.mean().item()

def test(model, test_dataloader):

    model.eval()
    total_dice = 0.
    total_iou  = 0.

    with torch.no_grad():  
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model(inputs)

            total_dice += dice_score(output, labels)
            total_iou  += iou_score(output, labels)

    mean_dice = total_dice / len(test_dataloader)
    mean_iou  = total_iou  / len(test_dataloader)

    print(f'Test Dice: {mean_dice:.4f}, Test IoU: {mean_iou:.4f}')

    return mean_dice, mean_iou


if __name__== "__main__":

    df = create_dataframe()
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)

    wandb.init(
    project="CT-scan-segmentation",
    name="evaluation",
    config={"threshold": 0.5}
    )

    test_ds  = CTLesionSegmentation(X_test,  y_test,  transform=val_test_transform)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(256, PATCH_SIZE, 1, EMBED_DIM, WIN, HEADS, SWIN_DEPTH, 1).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    mean_dice, mean_iou = test(model, test_dataloader)
    wandb.log({"test_dice": mean_dice, "test_iou": mean_iou})

    wandb.finish()