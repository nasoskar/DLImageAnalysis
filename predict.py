from train import UNet, create_dataframe, split_dataset
from dataset import CTLesionSegmentation, find_mean_std, transforms
from config import *
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
os.makedirs('prediction_images', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    df = create_dataframe()
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)
    mean_d, std_d = find_mean_std()
    train_transform, val_test_transform = transforms(mean_d, std_d)
    test_ds  = CTLesionSegmentation(X_test,  y_test,  transform=val_test_transform)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, EMBED_DIM, WIN, HEADS, SWIN_DEPTH, 1).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model_best_params_phase1.pth'))
    model.eval()

    images, masks = next(iter(test_dataloader))
    # just use first 4 images
    images = images[:4]
    masks  = masks[:4]

    with torch.no_grad():
        outputs = model(images.to(device))
        preds = (torch.sigmoid(outputs) > 0.5).float()  # binarise

    for i in range(4):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img  = images[i].cpu() * std + mean
        img  = img.permute(1,2,0).clamp(0,1).numpy()

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # top left — input
        axes[0][0].imshow(img)
        axes[0][0].set_title('Input')
        axes[0][0].axis('off')

        # top right — ground truth
        axes[0][1].imshow(masks[i].squeeze().cpu(), cmap='gray')
        axes[0][1].set_title('Ground truth')
        axes[0][1].axis('off')

        # bottom left — overlay
        overlay = img.copy()
        overlay[preds[i].squeeze().cpu().bool()] = [1, 0, 0]
        axes[1][0].imshow(overlay)
        axes[1][0].set_title('Overlay')
        axes[1][0].axis('off')

        # bottom right — prediction
        axes[1][1].imshow(preds[i].squeeze().cpu(), cmap='gray')
        axes[1][1].set_title('Prediction')
        axes[1][1].axis('off')

        plt.tight_layout()
        plt.savefig(f'prediction_images/predictions_sample_{i+1}.png')
        plt.close()