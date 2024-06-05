import torch
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import os

import pandas as pd
from sklearn.model_selection import train_test_split

import argparse

from guided_diffusion.train_util import Trainer
from guided_diffusion.save_log import create_folder
from guided_diffusion.script_util import (
    create_model_and_diffusion,
)

from isic_dataset import ISICDataset # ISICDatasetの前処理を自動でしてくれる

# main.pyの中身---------------------------------------------------------------------------------------------
# モデルアーキテクチャの引数等をここで一括管理できる
# argsにそれぞれの値をメモ書きする
# -------------------------------------------------------------------------------------------------------
def main(args):
    dir_path = os.path.join(args.path,"log",args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
    create_folder(dir_path)

    model,diffusion = create_model_and_diffusion(
        # modelの設定
        image_size=args.image_size,      # 画像サイズ
        in_channels =args.in_channels, # マスク画像のチャンネル数
        num_channels=128,              # モデル内でのデフォルトとなるチャンネル数
        out_channels=args.in_channels, # 出力画像のチャンネル数
        cond_img_ch=3,                 # 元画像のチャンネル数(条件付けのチャンネル数。元画像がカラーなので3)
        num_res_blocks=3,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16,8",  # アテンションをつける解像度。8,16はバランスが良い中間的な解像度。
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0,
        rrdb_blocks=2,
        deeper_net=True,
        
        # 拡散処理の設定
        steps=1000,         # 時間ステップ:T
        learn_sigma=False,  # 分散を学習するか
        sigma_small=False,
        noise_schedule="linear",  # ノイズのスケジュール
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="ddim100", # 何も指定しなければddpm, ddim100
        # timestep_respacing="ddim100", # 何も指定しなければddpm, ddim100
    )

    transform = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)), # 画像のリサイズ
        transforms.ToTensor()
        ])

# データセットを訓練用、検証用、テスト用にわける
    df = pd.read_csv(os.path.join(args.data_path, 'ISBI2016_ISIC_Part3B_' + "Training" + '_GroundTruth.csv'), encoding='gbk')
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed)
    test_df = pd.read_csv(os.path.join(args.data_path, 'ISBI2016_ISIC_Part3B_' + "Test" + '_GroundTruth.csv'), encoding='gbk')
        
    train_set = ISICDataset(args.data_path, train_df, transform=transform, mode='Training')
    val_set   = ISICDataset(args.data_path, val_df, transform=transform, mode='Training')
    test_set = ISICDataset(args.data_path, test_df, transform=transform, mode='Test')

    args.train_volume = len(train_set)
    args.val_volume = len(val_set)
    args.test_volum = len(test_set)
    

    optimizer = Adam(model.parameters(), lr=args.lr)
    
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        
        args=args,
        dir_path=dir_path,
        scheduler=None,
    )
    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser() # メモ作成する感じ
    
    # 以下にデータセットパス、サイズ、各パラメータを書く
    parser.add_argument('--path', type=str, default="/root/save")
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--data_path', type=str, default="/root/save/dataset/ISBI2016")
    parser.add_argument('--dataset', type=str, default='ISIC')
    parser.add_argument('--model_name', type=str, default='Segdiff')
    
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--in_channels', type=int, default=1)  # マスク画像のチャネル
    parser.add_argument('--cond_in_channels', type=int, default=3)  # 元画像のチャネル
      
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_n_model', type=int, default=50)
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--wandb_num_images', type=int, default=4)
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--project_name', type=str, default='aaa')  # プロジェクト名
    parser.add_argument('--model_detail', type=str, default='U-Netとか')  # ちょっとした詳細
    parser.add_argument('--train_volume', type=int, default=000)  # 訓練
    parser.add_argument('--val_volume', type=int, default=000)  # 検証
    parser.add_argument('--test_volume', type=int, default=000)  # テスト
    parser.add_argument('--img_size', type=str, default='32×32')  # 画像サイズ
    parser.add_argument('--optimizer', type=str, default='Adam')  # 最適化関数
    
    parser.add_argument('--dim_size', type=str, default='2D')
    args = parser.parse_args()
    main(args)