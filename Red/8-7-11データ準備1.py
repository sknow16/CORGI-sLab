import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# MNIST等の画像が入ってるデータベース
import torchvision.datasets as datasets

# ダウンロード先ディレクトリ名
data_root = './data'

train_set0 = datasets.MNIST(
    root = data_root,
    train = True,
    download = True)

# 正解データ付きで、最初の20個をイメージ表示
plt.figure(figsize=(10, 3))
for i in range(20):
    ax = plt.subplot(2, 10, i+1)
    
    # imageとlabelの取得
    image, label = train_set0[i]
    # イメージ表示
    plt.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('MNIST.png')
plt.show()


# データ件数の確認
print('データ件数：', len(train_set0))
# 最初の要素の取得
image, label = train_set0[0]
# データ型の確認
print('入力データの型：', type(image))
print('正解データの型：', type(label))

# 入力データの画像表示
plt.figure(figsize=(1,1))
# f'{}'で数字を文字扱いできる
plt.title(f'{label}') 
plt.imshow(image, cmap='gray_r')
# 画像表示の時にメモリを無くす
plt.axis('off') 