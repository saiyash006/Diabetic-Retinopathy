import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


train_tfms = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.5
    ),
    A.Normalize(),
    ToTensorV2()
])


val_tfms = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])


class RetinaDataset(Dataset):

    def __init__(self, csv_file, transforms=None):

        self.df = pd.read_csv(csv_file)

        self.transforms = transforms

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img = cv2.imread(row["image_path"])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = int(row["label"])

        if self.transforms:

            img = self.transforms(image=img)["image"]

        return img, torch.tensor(label)