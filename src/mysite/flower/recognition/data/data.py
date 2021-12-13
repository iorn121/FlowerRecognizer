import numpy as np
import cv2
from glob import glob
from torchvision import datasets, models, transforms
import os


def output_dataset_path_list(img_path, num_class=17, ratio=0.9):
    label_name = list(open('data/label.txt'))
    for i in range(len(label_name)):
        label_name[i] = label_name[i].replace('\n', '')
    train_data_list = []
    valid_data_list = []
    for i, name in enumerate(label_name):
        data_list = glob(f'data/images/{name}/*.jpg')
        select_idx = np.arrange(len(data_list))
        select_idx = np.random.choice(select_idx, int(
            len(data_list)*ratio), replace=False)
        for j, path in enumerate(data_list):
            if j in select_idx:
                train_data_list.append([path, i])
            else:
                valid_data_list.append([path, i])
            print(f'label name: {name}, ', end='')
            print(
                f'train: {len(select_idx)}, validation: {len(data_list)-len(select_idx)}')
    return train_data_list, valid_data_list


class MyDataset():
    def __init__(self, dataset_list, transform=None):
        self.dataset_list = dataset_list
        self.num_data = len(dataset_list)
        self.transform = transform

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        img = cv2.imread(self.dataset_list[idx][0])
        label = self.dataset_list[idx][1]

        # 0.5の確率で左右反転
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            out_data = self.transform(img)

        return out_data, label
