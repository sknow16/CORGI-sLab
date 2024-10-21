import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class ISICDataset(Dataset):
    def __init__(self, data_path, df, transform = None, mode = 'Training'):
        
        self.name_list = df.iloc[:, 0].tolist()  # 1番目の列が画像のファイル名（ISIC_0000000 形式）
        # self.label_list = df.iloc[:, 1].tolist()  # 2番目の列がラベル

        self.data_path = os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_Data')
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name + '.jpg')  # 画像ファイルのパス
        msk_path = os.path.join(self.data_path, name + '_Segmentation.png')  # セグメンテーションマスクのパス

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)
        
        # 学習のみパッチ分割
        if self.mode == "Training":
            patch_image = self.patch_image(img)
            patch_mask = self.patch_image(mask)
            return patch_image, patch_mask
        else:
            return (img, mask)

        # 学習とテストどちらもパッチ分割
        # patch_image = self.patch_image(img)
        # patch_mask = self.patch_image(mask)
        # return patch_image, patch_mask

    def patch_image(self, x):
        x1 = x[:, :128, :128]
        x2 = x[:, :128, 128:]
        x3 = x[:, 128:, :128]
        x4 = x[:, 128:, 128:]
        return x1, x2, x3, x4