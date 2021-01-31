import os
import sys
import timm
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from sklearn.model_selection import StratifiedKFold
import copy
import datetime


batch_size = 512
val_batch_size = 512
image_shape = (40, 40)


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

train_transform = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.1,p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5,sigma_limit=0.1),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.5),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.5),
    A.Resize(image_shape[0], image_shape[1]),
    # A.Cutout(max_h_size=int(img_image * 0.1), max_w_size=int(img_image * 0.1), num_holes=2, p=0.7),
    A.Normalize()
    ])

valid_transform = A.Compose([
    A.Resize(image_shape[0], image_shape[1]),
    A.Normalize()
])

test_transform = A.Compose([
    A.Resize(image_shape[0], image_shape[1]),
    A.Normalize()
])


# create dataset
class SarDataset(Dataset):
    def __init__(self, train_df, mode='train', transforms=None):
        self.train_df = train_df
        self.mode = mode
        self.transforms = transforms # 这里传过来的是一个变换

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        row = self.train_df.iloc[index]
        image_path, category = row.image_path, row.category
        image = cv2.imread(image_path, 1)

        if self.transforms:
            image = self.transforms(image=image)['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        image_tensor = torch.tensor(image).float()
        if self.mode == 'test':
            return image_tensor
        else:
            return image_tensor, torch.tensor(category).long()



def get_model(model_name, out_features, drop_rate=0.5):
    model = timm.create_model(model_name, pretrained=True)
    model.drop_rate = drop_rate
    model.classifier = nn.Linear(model.classifier.in_features, out_features)
    return model


def prepare_data(train_idx, val_idx, data_df):
    train_data = data_df.loc[train_idx, :].reset_index(drop=True)
    val_data = data_df.loc[val_idx, :].reset_index(drop=True)
    train_dataset = SarDataset(train_data, 'train', train_transform)
    # print(train_dataset.__len__())
    val_dataset = SarDataset(val_data, 'valid', valid_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader





