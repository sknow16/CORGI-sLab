import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from metric.metric import jaccard
import wandb
from tqdm.auto import tqdm
from isic_dataset import ISICDataset
import pandas as pd
from skimage.filters import threshold_otsu

data_path = "/root/save/dataset/ISBI2016"
train_df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + "Training" + '_GroundTruth.csv'), encoding='gbk')
test_df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + "Test" + '_GroundTruth.csv'), encoding='gbk')

train_len = len(train_df)
test_len = len(test_df)

print(train_len)
print(test_len)