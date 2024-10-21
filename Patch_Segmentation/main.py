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

class MNISTSegmentationDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.data = datasets.MNIST(
            root='/root/save/dataset', train=train, download=False, transform=transform
        )
        self.train = train
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image, label = self.data[idx]
        mask = (image > 0.5).float()  # 2値マスクを作成
        # セグメンテーションラベルを作成
        # segmentation_label = self.create_segmentation_label(image)
        if self.train:
            patch_image = self.patch_image(image)
            patch_mask = self.patch_image(mask)
            return patch_image, patch_mask
        else:
            return image, mask
    def patch_image(self, x):
        x1 = x[:, :128, :128]
        x2 = x[:, :128, 128:]
        x3 = x[:, 128:, :128]
        x4 = x[:, 128:, 128:]
        return x1, x2, x3, x4


# 畳み込み層、バッチ正規層、ReLU関数を各2回
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(   # nn.Sequential：モジュールを順序付けして1つのモジュールとして定義する
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):  # forward：純伝播処理
        y = self.convs(x)
        return y
# U-Net構造 ====================================================================
# time_embed_dim=100：各タイムステップに対応する特徴量の値(実際の時間とは関係ない)
# サイズ縮小：Maxプーリング　サイズ拡大：バイリニア補間
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128,256)
        self.bot1 = ConvBlock(256, 256)
        self.up3 = ConvBlock(256+256, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(128 + 64, 64)
        self.out = nn.Conv2d(64, out_ch, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        x3 = self.down3(x)
        x = self.maxpool(x3)
        x = self.bot1(x)
        x = self.upsample(x)
        x = torch.cat([x,x3], dim=1)
        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.out(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_path = "/root/save/dataset/ISBI2016"
train_df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + "Training" + '_GroundTruth.csv'), encoding='gbk')
test_df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + "Test" + '_GroundTruth.csv'), encoding='gbk')
        
# train, testのサイズを計算
train_size = len(train_df)
test_size  = len(test_df)
    
train_dataset = ISICDataset(data_path, train_df, transform=transform, mode='Training')
test_dataset = ISICDataset(data_path, test_df, transform=transform, mode='Test')
# train_dataset = MNISTSegmentationDataset(train=True, transform=transform)
# test_dataset = MNISTSegmentationDataset(train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
model = UNet(in_ch=3,out_ch=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

wandb.init(
   project = "Patch_UNet_ISIC2016",
   config ={
      "train_size": train_size,
      "test_size" : test_size,
   }
)
print(train_size, test_size)

for epoch in range(100):
  # 学習
  model.train()
  train_losses = []
  test_losses  = []
  iou_list     = []
  print(epoch+1,"epoch")
  for image, mask in tqdm(train_loader):
    for i in range(4):
      patch_image = image[i].to(device)
      patch_mask = mask[i].to(device)
      out = model(patch_image)
      optimizer.zero_grad()
      loss = nn.MSELoss()(out, patch_mask)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
  # 推論
  model.eval()
  for image, mask in tqdm(test_loader):
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
      out = model(image)
    loss = nn.MSELoss()(out, mask)

    # pred_mask = torch.sigmoid(out)
    # 閾値の変更
    # threshold = 0.5
    threshold = threshold_otsu(out.cpu().numpy())   # 大津閾値で決定
    pred_mask_binary = (out > threshold).float()

    iou = jaccard(pred_mask_binary.cpu().numpy(),mask.cpu().numpy())
    test_losses.append(loss.item())
    # iou_list.append(iou)
    iou_list.append(iou*len(mask))

  avg_iou = sum(iou_list)/test_size
  avg_trainloss = sum(train_losses)/len(train_losses)
  avg_testloss  = sum(test_losses)/len(test_losses)
  wandb_image = [wandb.Image(image[i]) for i in range(10)]
  wandb_mask  = [wandb.Image(mask[i]) for i in range(10)]
  wandb_pred  = [wandb.Image(pred_mask_binary[i]) for i in range(10)]
  wandb.log({
     "train_loss": avg_trainloss,
     "test_loss":  avg_testloss,
     "iou_loss":   avg_iou,
     "image": wandb_image,
     "mask": wandb_mask,
     "pred": wandb_pred
  })
