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
from shutil import copyfile

# hyper-parameters
train_dir = '/home/fcq/Competition/cvpr2021/train_images/'
val_dir = '/home/fcq/Competition/cvpr2021/val_images/'

# 先把train里面的eo图片和sar图像分开
train_sar = '/home/fcq/Competition/cvpr2021/train_sar/'
train_eo = '/home/fcq/Competition/cvpr2021/train_eo/'

if not os.path.exists(train_sar):
    os.mkdir(train_sar)
if not os.path.exists(train_eo):
    os.mkdir(train_eo)


for category in os.listdir(train_dir):
    if not os.path.exists(os.path.join(train_sar, category)):
        os.mkdir(os.path.join(train_sar, category))
    if not os.path.exists(os.path.join(train_eo, category)):
        os.mkdir(os.path.join(train_eo, category))
    for img in os.listdir(os.path.join(train_dir, category)):
        if img[:2] == 'SA':
            copyfile(os.path.join(train_dir, category, img), os.path.join(train_sar, category, img))
        else:
            copyfile(os.path.join(train_dir, category, img), os.path.join(train_eo, category, img))
    #     break
    # break
