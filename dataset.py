from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from albumentations.pytorch import ToTensorV2
from config import IMAGE_SIZE
import os
import albumentations as A
import numpy as np
import torch
import pandas as pd

def find_mean_std():
    img_dir = 'frames'
    means, stds = [], []
    for file in os.listdir(img_dir):
        if file.endswith('.png'):
            img = np.array(Image.open(os.path.join(img_dir, file)).convert('L')) / 255.0
            means.append(img.mean())
            stds.append(img.std())

    mean_d = np.mean(means)
    std_d  = np.mean(stds)

    return mean_d, std_d

def transforms(mean_d, std_d):
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE), #resize all images to have the same size 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=mean_d, std=std_d), #use the mean and std of the specific dataset
        #A.Normalize(mean=(0.485, 0.456, 0.406), #pre-trained weights
        #    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_test_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=mean_d, std=std_d),
        #A.Normalize(mean=(0.485, 0.456, 0.406), #pre-trained weights
        #    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_test_transform

class CTLesionSegmentation(Dataset):
    def __init__(self, images, masks, transform=None):
        self.data = images.reset_index(drop=True)
        self.masks = masks.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = Image.open(self.data[idx]).convert('L') # RGB for pre-trained weights 
        image_np = np.array(image)

        masks = Image.open(self.masks[idx]).convert('L')
        masks_np = np.array(masks)

        masks_np = masks_np / 255 #binarise the mask (0,1) to feed into the tensor
        
        if self.transform:
            augmented = self.transform(image = image_np, mask = masks_np)
            image_t = augmented['image']
            mask_t = augmented['mask'].unsqueeze(0)
        else:
            image_t = torch.from_numpy(image_np)
            mask_t = torch.from_numpy(masks_np)

        return image_t, mask_t
    

def create_dataframe():
    df = pd.DataFrame()

    training_folder = 'frames'
    groundtruth = 'masks'

    img_ids = [f.replace('.png', '') for f in os.listdir(training_folder) if f.endswith('.png')]

    df = pd.DataFrame({
        'feature': [os.path.join(training_folder, f'{i}.png') for i in img_ids],
        'target':  [os.path.join(groundtruth, f'{i}.png') for i in img_ids]
    })

    return df

def split_dataset(df):

    X_train, X_temp, y_train, y_temp = train_test_split(df['feature'], df['target'], train_size = 0.8, shuffle=True, random_state=42) 

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size = 0.5, shuffle=True, random_state=42) 

    return X_train, y_train, X_val, y_val, X_test, y_test
