import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# MNIST等の画像が入ってるデータベース
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ダウンロード先ディレクトリ名
data_root = './data'

transform = transforms.Compose([
    # データのテンソル化
    transforms.ToTensor(),
    # データの正規化
    transforms.Normalize(0.5, 0.5),
    # テンソルの1階テンソル化
    transforms.Lambda(lambda x: x.view(-1)),    
])

train_set0 = datasets.MNIST(
    root = data_root,
    train = True,
    download = True,
    transform = transform)

# 最初の要素の取得
image, label = train_set0[0]
# データ型の確認
print('入力データの型：', type(image))
print('正解データの型：', type(label))

