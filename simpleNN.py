import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

data_root = "./data" # ダウンロード先のディレクト名 (Google Colaboratoryの時は気にしなくていい)

# どのようなデータの前処理をするか指定
transform = transforms.Compose([
    transforms.ToTensor(),         # データをテンソル化処理 (画像をpytorchで扱えるtensor型に変形させる)
    transforms.Normalize(0.5,0.5), # データの正規化 範囲を[-1,1]に変更できる
])

# どのようにデータをダウンロードするか指定
test_set = datasets.MNIST(
    root  = data_root, # ど　こからデータをダウンロードするか
    train = False,      # 訓練データかテストデータか
    download  = True,  # 元データがない場合ダウンロードする
    transform = transform # 先ほど書いたデータの前処理をダウンロードするデータセットに行う
)

batch_size = 16 # ミニバッチのサイズ指定

# 訓練用データローダ― (データをダウンロードしてくれるやつ)
test_loader = DataLoader(
    test_set, # データの処理方法を指定
    batch_size = batch_size,
    shuffle    = True # データをシャッフルするかどうか
)

for images, label in test_loader:
  break
# imagesの先頭を取り出す
# imagesにはバッチの数の画像が含まれている
print("imagesの長さ:",len(images))
img = images[0]
plt.imshow(img.permute(1,2,0),cmap="gray")
plt.show()