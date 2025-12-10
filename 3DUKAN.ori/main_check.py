from dataset import brain_dataset
from model import UNet3D
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    Resized,
    ToTensord,
    MapTransform,
    ConvertToMultiChannelBasedOnBratsClassesD,
    Spacingd,
)
from tqdm.auto import tqdm
import argparse
from sklearn.model_selection import train_test_split
from train_util import Trainer
from torch.optim import Adam
import logging
from metric.metric import dice
from torch.utils.data import DataLoader

def main(args):
    df = pd.read_csv(args.csv_path)
    # 画像の前処理 (今は、ラベルの情報が0,1,2,3になっていたのがResizeでおかしくなってることに注意)
    transform = Compose([
        # ラベルを3から4に変換

        ConvertToMultiChannelBasedOnBratsClassesD(keys="label"),
        # Resized(keys=["image"], spatial_size=(args.image_size, args.image_size, args.volume_size), mode="trilinear"),  # 画像のサイズ変更
        # Resized(keys=["label"], spatial_size=(args.image_size, args.image_size, args.volume_size), mode="nearest"),  # ラベルのサイズ変更
        ToTensord(keys=["image", "label"]),
    ])


    train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    test_df, val_df = train_test_split(val_test_df, test_size=0.5, random_state=args.seed)


    args.train_size, args.val_size, args.test_size = len(train_df), len(val_df), len(test_df)
    
    train_set = brain_dataset(train_df, transform=transform)
    val_set   = brain_dataset(val_df, transform=transform)
    test_set  = brain_dataset(test_df, transform=transform)
    model     = UNet3D(args.in_channels, args.out_channels)
    optimizer = Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.MSELoss()

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=True)

    batch = train_set[32]
    image, label = batch['image'], batch['label']
    print(image.shape)
    print(label.shape)
    # labelのunque値を確認
    print(torch.unique(label))
    # 3を4に変換
    # labelの可視化(63のスライス)

    # plt.title('Image Slice')
    # 軸なし
    # plt.axis('off')
    # plt.imshow(label[0,:,:,63], cmap='gray')
    # plt.savefig(
    #     'check_image_channel.png',
    #     bbox_inches='tight',  # 内容に合わせてトリミング
    #     pad_inches=0,         # 余白ゼロ
    # )
    plt.axis('off')
    plt.imshow(image[0,:,:,63], cmap='gray')
    plt.savefig('check_image_channel0.png', bbox_inches='tight', pad_inches=0)
    plt.imshow(image[1,:,:,63], cmap='gray')
    plt.savefig('check_image_channel1.png', bbox_inches='tight', pad_inches=0)
    plt.imshow(image[2,:,:,63], cmap='gray')
    plt.savefig('check_image_channel2.png', bbox_inches='tight', pad_inches=0)
    plt.imshow(image[3,:,:,63], cmap='gray')
    plt.savefig('check_image_channel3.png', bbox_inches='tight', pad_inches=0)

    plt.imshow(label[0,:,:,63], cmap='gray')
    plt.savefig('check_label_channel0.png', bbox_inches='tight', pad_inches=0)
    plt.imshow(label[1,:,:,63], cmap='gray')
    plt.savefig('check_label_channel1.png', bbox_inches='tight', pad_inches=0)
    plt.imshow(label[2,:,:,63], cmap='gray')
    plt.savefig('check_label_channel2.png', bbox_inches='tight', pad_inches=0)



   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=40) # 今まで42
    parser.add_argument('--csv_path', type=str, default='/root/save/dataset/Task01_BrainTumour/id_list.csv')

    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=4)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=3)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='3DU-Net_Brats_zmini_2025')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='3DU-Net_128_128_64')
    parser.add_argument('--dataset', type=str, default='Brats')
    parser.add_argument('--model_detail', type=str, default='論文まねした。')   # ちょっとした詳細

    parser.add_argument('--train_size', type=int, default=0)  # 訓練
    parser.add_argument('--val_size', type=int, default=0)    # 検証
    parser.add_argument('--test_size', type=int, default=0)   # テスト

    parser.add_argument('--img_size', type=str, default='128×128×64')       # 画像サイズ (wandbに送るよう)
    parser.add_argument('--optimizer', type=str, default='Adam')         # 最適化関数 (wandbに送るよう)
    parser.add_argument('--loss_function', type=str, default='MSELoss')  # 損失関数 (wandbに送るよう)

    parser.add_argument('--dim_size', type=str, default='3D')       # 2D or 3D (wandbに送るよう)
    args = parser.parse_args()
    main(args)