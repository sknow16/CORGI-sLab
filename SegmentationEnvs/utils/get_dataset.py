import pandas as pd
import os
import torch.nn.functional as F
import torchvision
import os
import torch
from torchvision.transforms import InterpolationMode

from utils.get_transform import get_transform

def get_dataset(cfg):  
    transform, val_transform = get_transform(cfg)
    if cfg["data"]['dataset']== "WDDD2":
        if cfg['task'] == 'pos':
            from dataset import WDDD2Dataset
            train_set   = WDDD2Dataset(csv_path=cfg["data"]["pos"]["train_csv_path"], transform=transform)
            test_set    = WDDD2Dataset(csv_path=cfg["data"]["pos"]["test_csv_path"], transform=val_transform)
            val_set     = test_set
        elif cfg["task"] == 'seg':
            from dataset import SegWDDD2Dataset
            train_set = SegWDDD2Dataset(csv_path=cfg["data"]["seg"]["train_csv_path"], transform=transform)
            val_set   = SegWDDD2Dataset(csv_path=cfg["data"]["seg"]["val_csv_path"], transform=val_transform)
            test_set  = SegWDDD2Dataset(csv_path=cfg["data"]["seg"]["test_csv_path"], transform=val_transform)

    return train_set, val_set, test_set