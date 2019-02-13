import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

img_size = 225
random_crop_r = 5
lfw_dir = '../../../../Courses_data'
lfw_dataset_dir = '../../../../Courses_data/lfw'
train_set_path = os.path.join(lfw_dir, 'LFW_annotation_train.txt')
test_set_path = os.path.join(lfw_dir, 'LFW_annotation_test.txt')

torch.set_default_tensor_type('torch.cuda.FloatTensor')
# get list
def get_list(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line in ['\n', '\r\n']:
                result = line.split('\t')
                img_name = result[0]
                bbox_list = result[1].split()
                bbox = np.array(bbox_list).astype(np.int).reshape((-1,2))
                landmarks_list = result[2].split()
                landmarks = np.array(landmarks_list).astype(np.float).reshape((-1,2))
                data_list.append({'name': img_name, 'bbox': bbox, 'lm': landmarks})
    return data_list

def show_landmarks(img, landmarks):
    length = len(img)
    figs, axes = plt.subplots(1, length)
    for i in range(0, length):
        axes[i].imshow((img[i] + 1) / 2)
        axes[i].set_title('Sample:' + str(i))
        axes[i].scatter(landmarks[i][0]*img_size, landmarks[i][1]*img_size, s=20, marker='.', c='r')
    plt.pause(0.01)  # pause a bit so that plots are updated
    plt.show()

class LFWDataSet(Dataset):
    def __init__(self, data_list, transform=[]):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_name = item['name']
        bbox = item['bbox']
        landmarks = item['lm']
        img_path = img_name.split('.')[0][0:-5]
        file_path = os.path.join(lfw_dataset_dir, img_path, img_name)

        # img = np.array(Image.open(file_path), dtype=np.float) / 255.0 * 2 - 1
        img = Image.open(file_path)
        w, h = bbox[1] - bbox[0]

        # random crop
        if 'rcrop' in self.transform:
            if random.random() < 0.5:
                top = np.random.randint(0, int(landmarks[:,0].min() - bbox[0][0])+1) - random_crop_r
                left = np.random.randint(0, int(landmarks[:,1].min() - bbox[0][1])+1) - random_crop_r
                offset = [[top,left],[top,left]]
                bbox += offset
        img = img.crop((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))
        img = img.resize((img_size, img_size))
        landmarks_ = []
        for coor in landmarks:
            coor = coor - bbox[0]
            coor[0] = coor[0] / w  # 0~225
            coor[1] = coor[1] / h 
            landmarks_.append(coor)
        landmarks_ = np.array(landmarks_, dtype=np.float).T
        
        # random flip
        if 'flip' in self.transform:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                landmarks_[0] = np.ones(7) - landmarks_[0]
                landmarks_ = landmarks_[:,[3,2,1,0,5,4,6]] # change order
        
        img = np.array(img, dtype=np.float) / 255.0 * 2 - 1
        img_tensor = torch.from_numpy(img).float()
        landmarks_tensor = torch.from_numpy(landmarks_).float()

        return img_tensor, landmarks_tensor



# test for original img and landmarks
# test_list = test_list[0: 4]
# figs, axes = plt.subplots(1, 4)
# for i in range(0, 4):
#     item = test_list[i]
#     img_name = item['name']
#     bbox = item['bbox']
#     landmarks = item['lm']
#     img_path = img_name.split('.')[0][0:-5]
#     file_path = os.path.join(lfw_dataset_dir, img_path, img_name)
#     img = np.array(Image.open(file_path), dtype=np.float) / 255
#     axes[i].imshow(img)
#     landmarks = np.array(landmarks).T
#     axes[i].scatter(landmarks[0], landmarks[1], s=20, marker='.', c='r')
#     axes[i].set_title('Sample:' + str(i))
# plt.pause(0.01)
# plt.show()