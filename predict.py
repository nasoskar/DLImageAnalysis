from train import UNet, create_dataframe, split_dataset
from dataset import CTLesionSegmentation, val_test_transform
from config import *
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    df = create_dataframe()
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)
    test_ds  = CTLesionSegmentation(X_test,  y_test,  transform=val_test_transform)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(256, PATCH_SIZE, 1, EMBED_DIM, WIN, HEADS, SWIN_DEPTH, 1).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    images, masks = next(iter(test_dataloader))
    # just use first 4 images
    images = images[:4]
    masks  = masks[:4]

    with torch.no_grad():
        outputs = model(images.to(device))
        preds = (torch.sigmoid(outputs) > 0.5).float()  # binarise

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img  = images[i] * std + mean  # denormalise
        img  = img.clamp(0,1)
        # original image
        img = images[i].cpu() * std + mean
        axes[i][0].imshow(img.permute(1,2,0).clamp(0,1))
        axes[i][0].set_title('Input')

        # ground truth
        axes[i][1].imshow(masks[i].squeeze().cpu(), cmap='gray')
        axes[i][1].set_title('Ground truth')

        # prediction
        axes[i][2].imshow(preds[i].squeeze().cpu(), cmap='gray')
        axes[i][2].set_title('Prediction')

        # overlay
        img_np = img.permute(1,2,0).clamp(0,1).numpy()
        overlay = img_np.copy()
        overlay[preds[i].squeeze().cpu().bool()] = [1, 0, 0]  # red for predicted lesion
        axes[i][3].imshow(overlay)
        axes[i][3].set_title('Overlay')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()