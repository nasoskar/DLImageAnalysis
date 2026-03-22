from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from albumentations.pytorch import ToTensorV2

import os
import albumentations as A
import numpy as np
import torch
import pandas as pd


train_transform = A.Compose([
    A.Resize(256, 256), #resize all images to have the same size 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), #use the mean and std of ImageNet when using a pre-trained encoder
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class SkinLesionSegmentation(Dataset):
    def __init__(self, images, masks, transform=None):
        self.data = images.reset_index(drop=True)
        self.masks = masks.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = Image.open(self.data[idx]).convert('RGB')
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

    training_folder = 'ISIC2018_Task1-2_Training_Input'
    groundtruth = 'ISIC2018_Task1_Training_GroundTruth'

    img_ids = [f.replace('.jpg', '') for f in os.listdir(training_folder) if f.endswith('.jpg')]

    df = pd.DataFrame({
        'feature': [os.path.join(training_folder, f'{i}.jpg') for i in img_ids],
        'target':  [os.path.join(groundtruth, f'{i}_segmentation.png') for i in img_ids]
    })

    return df

def split_dataset(df):

    X_train, X_temp, y_train, y_temp = train_test_split(df['feature'], df['target'], train_size = 0.8, shuffle=True, random_state=42) 

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size = 0.5, shuffle=True, random_state=42) 

    return X_train, y_train, X_val, y_val, X_test, y_test


#---PROCESS---
#Step 1: create dataframe based on the image paths
df = create_dataframe()

#Step 2: Split the dataset
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)

#Step 3: Create Dataset objects
train_ds = SkinLesionSegmentation(X_train, y_train, transform=train_transform)
val_ds   = SkinLesionSegmentation(X_val,   y_val,   transform=val_test_transform)
test_ds  = SkinLesionSegmentation(X_test,  y_test,  transform=val_test_transform)

#Step 4:  Call DataLoader
train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=16, shuffle=False)

#for data in train_dataloader:
#    print(data)
images, masks = next(iter(train_dataloader))

print(f'Image shape: {images.shape}')   # expect (16, 3, 256, 256)
print(f'Mask shape:  {masks.shape}')    # expect (16, 1, 256, 256)
print(f'Image range: {images.min():.2f} to {images.max():.2f}')
print(f'Mask values: {masks.unique()}') # expect tensor([0., 1.])