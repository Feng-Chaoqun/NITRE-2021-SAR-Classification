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

# hyper-parameters
train_dir = '/home/fcq/Competition/cvpr2021/train_sar/'

checkpoint_dir = ''
result = ''
epoch = 10
model_name = 'tf_efficientnet_b4'
class_num = 10
device = 'cuda'
batch_size = 512
val_batch_size = 512
image_shape = (40, 40)
epoches = 30
experience = False
loss_record = pd.DataFrame(columns = ['epoch', 'train_loss', 'val_loss', 'epoch_lr'])

def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
# make dataset
train_data =[]
for category in os.listdir(train_dir):
    for image in os.listdir(os.path.join(train_dir, category)):
        train_data.append([os.path.join(train_dir, category, image), int(category)])

train_df = pd.DataFrame(train_data, columns=['image_path', 'category'], index=None)

# transform
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

current_model = get_model(model_name, 10)
current_model.to(device)

optimizer = optim.Adam(current_model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=epoches)
criterion = nn.CrossEntropyLoss()

def train_val(model, optimizer, criterion, train_loader, val_loader, train_size, val_size):
    train_loss_total_epoches, valid_loss_total_epoches, epoch_lr = [], [], []
    best_loss = 1e50
    best_model = copy.deepcopy(model)
    start = datetime.datetime.now()
    for epoch in range(epoches):
        print('run No.{} epoches, lr = {}'.format(epoch,optimizer.param_groups[0]['lr']))
        model.train()
        train_loss_per_epoch = 0
        for batch_index, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            loss_np = loss.detach().cpu().numpy()
            train_loss_per_epoch += loss_np
            print('Epoch:{}, Batch_index:{}, Training Loss:{:.8f},'.format(epoch, batch_index, loss_np))

        average_train_loss = train_loss_per_epoch / train_size

        model.eval()
        valid_loss_per_epoch = 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pred = model(data)
                loss = criterion(pred, label)
                loss_np = loss.detach().cpu().numpy()
                valid_loss_per_epoch += loss_np
            average_val_loss = valid_loss_per_epoch / val_size

        train_loss_total_epoches.append(average_train_loss)
        valid_loss_total_epoches.append(average_val_loss)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        loss_record.loc[epoch, :] = [epoch, average_train_loss, average_val_loss, optimizer.param_groups[0]['lr']]

        # save model
        if epoch % 1 == 0:
            state = {'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False)
        if average_val_loss < best_loss:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
            torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False)
            best_loss = average_val_loss
            best_model = copy.deepcopy(model)
        scheduler.step()

        time_consume = datetime.datetime.now() -start
        if epoch % 1 == 0:
                    print('Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}, Time Consuming:{}'.format(epoch, average_train_loss, average_val_loss, time_consume))




    # 训练loss曲线
    if True:
        x = [i for i in range(epoches)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epoches, 0.6), label='训练集loss')
        ax.plot(x, smooth(valid_loss_total_epoches, 0.6), label='验证集loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title(f'训练曲线', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'学习率变化曲线', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

def prepare_data(train_idx, val_idx, data_df):
    train_data = data_df.loc[train_idx, :].reset_index(drop=True)
    val_data = data_df.loc[val_idx, :].reset_index(drop=True)
    train_dataset = SarDataset(train_data, 'train', train_transform)
    # print(train_dataset.__len__())
    val_dataset = SarDataset(val_data, 'valid', valid_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

k_fold = StratifiedKFold(n_splits=5, shuffle=False)
for train_index, val_index in k_fold.split(train_df.image_path, train_df.category):
    train_loader, val_loader = prepare_data(train_index, val_index, train_df)
    train_size, val_size = len(train_index), len(val_index)
    train_val(current_model, optimizer, criterion, train_loader,val_loader, train_size, val_size)
    break





