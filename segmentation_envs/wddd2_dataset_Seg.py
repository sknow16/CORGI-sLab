# 線虫WDDD2のデータセットを作るコード

import tifffile
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

class SegWDDD2Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.data_path = df.iloc[:,0].tolist()  # csvファイルのid(カルム)を抜いた1列目のパスをリストに入れる ⇒ WDDD2のパスがリストに入る
        self.transform = transform

    def __len__(self):
        return len(self.data_path)
    
    # 画素値が0,1,2のマスク画像を2チャネルの0,1のマスク画像に変更
    def separete_channel(self, mask):
        # マスクをNumPy配列に変換
        mask_array = np.array(mask)  # PIL.Image -> NumPy array

        # マスクを2チャネル化
        channel0 = (mask_array == 0).astype(np.float32)  # 値1をチャネル1に
        channel1 = (mask_array == 1).astype(np.float32)  # 値1をチャネル1に
        channel2 = ((mask_array == 1)| (mask_array == 2)).astype(np.float32)  # 値2をチャネル2に

        mask_2channel = np.stack([channel0, channel1, channel2], axis=0)  # [2, H, W]

        # PyTorchのテンソルに変換
        mask_2channel = torch.from_numpy(mask_2channel)
        return mask_2channel
    
    def __getitem__(self, index):
        """Get the images"""
        path= self.data_path[index]  # index ⇒ data_pathの任意の列
        image_path = os.path.join('/root/save/dataset/WDDD2_2D_seg/annotation_image/images', path) + '.tiff'
        mask_path = os.path.join('/root/save/dataset/WDDD2_2D_seg/annotation_image/annotation', path) + '.png'

        img = Image.open(image_path)    
        mask = Image.open(mask_path)
        mask = self.separete_channel(mask) # チャネルを２チャネルに変更
        if self.transform:
            # トランスフォームの適用
            state = torch.get_rng_state()
            img = self.transform["image"](img)  # 画像にトランスフォーム
            torch.set_rng_state(state)
            mask = self.transform["mask"](mask)
        return img, mask

# from torchvision.transforms import InterpolationMode
# # 2. トランスフォームの定義
# transform = {
#         "image": transforms.Compose([
#             transforms.Resize((128,128), interpolation=InterpolationMode.BILINEAR), 
#             transforms.ToTensor()  # Tensorに変換
#             ]),
#         "mask": transforms.Compose([
#             transforms.Resize((128,128), interpolation=InterpolationMode.NEAREST),  
#             # transforms.ToTensor()  # Tensorに変換
#             ]),
#     }
# # 3. データセットのインスタンス化
# dataset = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segtrain.csv', transform=transform)

# # 4. データセットの長さを確認
# print(f"Dataset size: {len(dataset)}")

# # 6. DataLoaderを使用して、データをバッチ単位で読み込む
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # バッチごとの画像を確認
# for batch in dataloader:
#     print(f"Batch shape: {batch[0].shape}")  # [batch_size, 3, 128, 128] のはず
#     print(f"Batch shape: {batch[1].shape}")  # [batch_size, 3, 128, 128] のはず

#     mask = batch[1][0][0]
#     import matplotlib.pyplot as plt
#     plt.imshow(mask)
#     plt.savefig("test_mask_ch0.png")
#     plt.close()
#     mask = batch[1][0][1]
#     import matplotlib.pyplot as plt
#     plt.imshow(mask)
#     plt.savefig("test_mask_ch1.png")
#     plt.close()
#     mask = batch[1][0][2]
#     import matplotlib.pyplot as plt
#     plt.imshow(mask)
#     plt.savefig("test_mask_ch2.png")
#     plt.close()
#     print(mask.shape)
#     flat = mask.flatten()
#     print(flat.shape)
#     box = []
#     for xi in flat:
#         if xi not in box:
#             box.append(xi)
#     print(box)
#     break

