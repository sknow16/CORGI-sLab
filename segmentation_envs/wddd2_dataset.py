# 線虫WDDD2のデータセットを作るコード

import tifffile
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms

class WDDD2Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.data_path = df.iloc[:,0].tolist()  # csvファイルのid(カルム)を抜いた1列目のパスをリストに入れる ⇒ WDDD2のパスがリストに入る
        self.transform = transform

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        """Get the images"""
        path= self.data_path[index]  # index ⇒ data_pathの任意の列
        img = Image.open(path)      
        if self.transform:
            img = self.transform(img)
            
        return img


# # 2. トランスフォームの定義
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # 画像を128x128にリサイズ
#     transforms.ToTensor()  # Tensorに変換
# ])

# # 3. データセットのインスタンス化
# dataset = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftrain.csv', transform=transform)

# # 4. データセットの長さを確認
# print(f"Dataset size: {len(dataset)}")

# # 5. 一つの画像を取得して確認
# image = dataset[0]
# print(f"Image shape: {image.shape}")  # [3, 128, 128] になるはず

# # 6. DataLoaderを使用して、データをバッチ単位で読み込む
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # バッチごとの画像を確認
# for batch in dataloader:
#     print(f"Batch shape: {batch.shape}")  # [batch_size, 3, 128, 128] のはず