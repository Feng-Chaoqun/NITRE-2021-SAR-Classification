import torch
import os
import pandas as pd
import numpy as np
from my_image_classification.data import test_transform
from torch.utils.data import DataLoader
import cv2
import torch
import timm
import torch.nn as nn



def get_model(model_name, out_features, drop_rate=0.5):
    model = timm.create_model(model_name, pretrained=False)
    model.drop_rate = drop_rate
    model.classifier = nn.Linear(model.classifier.in_features, out_features)
    return model

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
test_dir = '/home/fcq/Competition/cvpr2021/val_images/'
checkpoint = '/home/fcq/Competition/nitre_baseline/my_image_classification/checkpoint-epoch28.pth'
model_name = 'tf_efficientnet_b4'
model = get_model(model_name, 10)
model.load_state_dict(torch.load(checkpoint)['state_dict'])
model.to('cuda')
model.eval()

test_data =[]
for image in os.listdir(test_dir):
    id = image[:-4].split('_')[1]
    test_data.append([os.path.join(test_dir, image), int(id), 0])

test_df = pd.DataFrame(test_data, columns=['image_path', 'image_id', 'class_id'], index=None)

with torch.no_grad():
    for index in range(len(test_df)):
        image = cv2.imread(test_df.iloc[index].image_path)
        image = test_transform(image=image)['image'].astype(np.float32)
        image = image.transpose(2, 0, 1)
        image_tensor = torch.tensor(image).unsqueeze(0).float().cuda()
        pred = model(image_tensor)
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.detach().cpu().numpy()
        label = pred.argmax()
        print(label)
        test_df.loc[index, 'class_id'] = [label]

test_df.loc[:, ['image_id', 'class_id']].to_csv('/home/fcq/Competition/cvpr2021/result/result_efficientnet_no_sofmax.csv', index=False)






# test_dataset = SarDataset(test_data, 'test', test_transform)
# test_loader = DataLoader(test, batch_size=val_batch_size, shuffle=False, num_workers=4)
# def pred(model, data):
#     model.eval()
#     device = torch.device("cuda")
#     data = data.transpose(2, 0, 1)
#     if data.max() > 1: data = data / 255
#     c, x, y = data.shape
#     label = np.zeros((x, y))
#     x_num = (x//target_l + 1) if x%target_l else x//target_l
#     y_num = (y//target_l + 1) if y%target_l else y//target_l
#     for i in tqdm(range(x_num)):
#         for j in range(y_num):
#             x_s, x_e = i*target_l, (i+1)*target_l
#             y_s, y_e = j*target_l, (j+1)*target_l
#             img = data[:, x_s:x_e, y_s:y_e]
#             img = img[np.newaxis, :, :, :].astype(np.float32)
#             img = torch.from_numpy(img)
#             img = Variable(img.to(device))
#             out_l = model(img)
#             out_l = out_l.cpu().data.numpy()
#             out_l = np.argmax(out_l, axis=1)[0]
#             label[x_s:x_e, y_s:y_e] = out_l.astype(np.int8)
#     print(label.shape)
#     return label