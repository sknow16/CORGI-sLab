from wddd2_dataset import WDDD2Dataset
from model import UNet
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import argparse
from sklearn.model_selection import train_test_split
from train_util_copy import Trainer
from torch.optim import Adam
from guided_diffusion.save_log import create_folder
import os
from guided_diffusion.script_util import create_gaussian_diffusion
import matplotlib.pyplot as plt
import random

x = torch.randn(1,1,32,32)
mask = torch.zeros(8,8)

def random_coordinate(mask_point_range, batch):
    coordinates = [(random.randint(0, mask_point_range), random.randint(0, mask_point_range)) for _ in range(batch)]
    return coordinates

batch = 1
mask_point_range = 24

# random_coordinates = random_coordinate(mask_point_range, batch)
# print(random_coordinates)

def create_crop(x, crop_size):
    n, c, h, w = x.shape
    left_points = random_coordinate(w-crop_size, n)
    mini_x = torch.zeros(n, c, crop_size, crop_size)
    for i in range(n):
        mini_x[i] = x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]
        # x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size] = 0
        x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size] = torch.clamp(x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]+0.3, 0, 1)
    return mini_x, x

def main(args):
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  
        transforms.ToTensor()  # Tensorに変換
    ])
    Ftraindataset   = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftrain.csv', transform=transform)
    train_loader = DataLoader(Ftraindataset, batch_size=16, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=False)
   # 画素値を蓄積するリスト
    all_pixels = []

    # すべての画像を収集
    for images in tqdm(train_loader):  # train_loaderからバッチを取得
        x, cond = create_crop(images.clone(), crop_size=64)
        # images = images.numpy()  # Tensorをnumpyに変換
        break
        # 3チャネルに
        

        all_pixels.append(images.flatten())  # バッチ内のすべての画素を1次元配列として追加
    
    
    # second = images[0].permute(1,2,0)
    print(images.min())
    plt.imshow(images[0].permute(1,2,0), cmap="gray")
    plt.savefig('./test.png')
    plt.close()

    plt.imshow(cond[0].permute(1,2,0), cmap="gray")
    plt.savefig('./test2.png')
    plt.close()
    # # 全画像の画素値を1次元配列に統合
    # all_pixels = np.concatenate(all_pixels)
    print(images.shape)
    print(cond.shape)
    images = images[0].flatten()
    images = images.numpy()
    cond = cond[0].flatten()
    cond = cond.numpy()

    # 最初のヒストグラム（images）
    plt.hist(images, bins=100, color='blue', alpha=0.5, label='Images', range=(0, 1))

    # 2つ目のヒストグラム（cond）
    plt.hist(cond, bins=100, color='red', alpha=0.5, label='Conditioned', range=(0, 1))

    # グラフのタイトルとラベル
    plt.title('Histogram of All Pixels')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    # 凡例を表示
    plt.legend()
    # グリッドを表示
    plt.grid(True)
    # 保存と表示
    plt.savefig("./hist_combined.png")
    # plt.figure(figsize=(10, 6))
    # plt.hist(images, bins=100, color='blue', alpha=0.7, range=(0,1))
    # plt.title('Histogram of All Pixels')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig("./hist.png")
    # # ヒストグラムをプロット
    # plt.figure(figsize=(10, 6))
    # plt.hist(cond, bins=100, color='red', alpha=0.7, range=(0,1))
    # plt.title('Histogram of All Pixels')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig("./hist2.png")
    # plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, default="/root/save")

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=3)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=3)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='crop_wddd2')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='masked_gen')
    parser.add_argument('--dataset', type=str, default='Brats')
    parser.add_argument('--model_detail', type=str, default='2D画像の一部をマスクして再構成する')   # ちょっとした詳細

    parser.add_argument('--train_size', type=int, default=0)  # 訓練
    parser.add_argument('--val_size', type=int, default=0)    # 検証
    parser.add_argument('--test_size', type=int, default=0)   # テスト

    parser.add_argument('--img_size', type=str, default='256×256')       # 画像サイズ (wandbに送るよう)
    parser.add_argument('--optimizer', type=str, default='Adam')         # 最適化関数 (wandbに送るよう)
    parser.add_argument('--loss_function', type=str, default='MSELoss')  # 損失関数 (wandbに送るよう)

    parser.add_argument('--dim_size', type=str, default='2D')       # 2D or 3D (wandbに送るよう)
    args = parser.parse_args()
    main(args)