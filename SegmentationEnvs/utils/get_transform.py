import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_transform(cfg):
    if cfg["data"]["dataset"] == "WDDD2":
        transform = {
            "image": transforms.Compose([
                transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"]), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5), # y軸方向を中心に反転
                transforms.RandomVerticalFlip(p=0.5),   # x軸方向を中心に反転
                transforms.ToTensor(),  # Tensorに変換
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
            "mask": transforms.Compose([
                transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"]), interpolation=InterpolationMode.NEAREST),  
                transforms.RandomHorizontalFlip(p=0.5), # y軸方向を中心に反転
                transforms.RandomVerticalFlip(p=0.5),   # x軸方向を中心に反転
                ]),
        }

        val_transform = {
            "image": transforms.Compose([
                transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"]), interpolation=InterpolationMode.BILINEAR), 
                transforms.ToTensor(),  # Tensorに変換
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
            "mask": transforms.Compose([
                transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"]), interpolation=InterpolationMode.NEAREST),  
                ]),
        }
    return transform, val_transform
