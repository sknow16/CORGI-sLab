import os
import random

import SimpleITK as sitk
import numpy as np
from monai import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from monai.transforms import (
    Compose,
    Resized,
    ToTensord,
    MapTransform,
    ConvertToMultiChannelBasedOnBratsClassesD,
    Spacingd,
)

class preDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist

    def read_data(self, data_path):
        # file_identifizer = data_path.split("/")[-1].split("-")[-2]
        file_idx = data_path.split("_")[-1]
        image_paths = [
            os.path.join(data_path, f"BraTS2021_{file_idx}_t1.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_idx}_t1ce.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_idx}_t2.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_idx}_flair.nii.gz")
        ]

        seg_path = os.path.join(data_path, f"BraTS2021_{file_idx}_seg.nii.gz")

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

        np.place(seg_data, seg_data == 3, 4)

        image_data = np.array(image_data).astype(np.float32)
        return [image_data, seg_data[np.newaxis, :, :, :], f"BraTS2021_{file_idx}"]

    def __getitem__(self, i):
        img, tmp, path = self.read_data(self.datalist[i])
        image = {"image": img, "label": tmp}
        if self.transform is not None:
            image = self.transform(image)
        image["path"] = path
        return image

    def __len__(self):
        return len(self.datalist)
    
# dir_path = "/root/save/dataset/Brats2021/BraTS2021_Training_Data"
# all_data_list = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]   

# transform = Compose([
#         ConvertToMultiChannelBasedOnBratsClassesD(keys="label"),
#         Resized(keys=["image"], spatial_size=(255, 255, 255), mode="trilinear"),  # 画像のサイズ変更
#         Resized(keys=["label"], spatial_size=(255, 255, 255), mode="nearest"),  # ラベルのサイズ変更
#         ToTensord(keys=["image", "label"]),
#     ])
# dataset = preDataset(all_data_list, transform=transform)
# print(f"データ数: {len(dataset)}")    

# image = dataset[0]["image"]
# label = dataset[0]["label"]
# print(f"画像サイズ: {image.shape}")  # 4チャネルの画像データ
# print(f"ラベルサイズ: {label.shape}")  # セグメンテーションマスク
# print(f"ラベルのユニークな値: {np.unique(label)}")  # ラベルのユニークな値
# print(f"データパス: {dataset[0]['path']}")

