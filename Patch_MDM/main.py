"""
Train a diffusion model on images.
"""
import os
import torch
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
sys.path.append("..")
sys.path.append(".")
from torchvision import transforms
from torch.optim import Adam
from guided_diffusion.wddd2_dataset import WDDD2Dataset

from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model,
    create_gaussian_diffusion,
)
from guided_diffusion.train_util import Trainer
from guided_diffusion.save_log import create_folder

def main(args):
    dir_path = os.path.join(args.path,"log",args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
    create_folder(dir_path)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  
        transforms.ToTensor()  # Tensorに変換
    ])
    Ftraindataset   = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftrain.csv', transform=transform)
    Ftestdataset    = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftest.csv', transform=transform)
    model     = create_model(image_size=args.image_size,
                             num_channels=32,
                             num_res_blocks=2,
                             in_channels=args.in_channels
                             )
    diffusion = create_gaussian_diffusion(in_channels=args.in_channels,
                                          steps=1000,
                                          patch_size=8

                                          )
    optimizer = Adam(model.parameters(),lr=args.lr)
    args.val_volume = len(Ftestdataset)
    args.train_volume = len(Ftraindataset)
    trainer = Trainer(
        model=model,
        diffusion = diffusion,
        optimizer=optimizer,
        train_set=Ftraindataset,
        val_set=Ftestdataset,
        # test_set=test_set,
        args=args,
        dir_path=dir_path,
    )
    trainer.train()
    # import torch
    # print(torch.version.cuda)  # インストールされているCUDAバージョンを確認
    # print(torch.cuda.is_available())  # CUDAが利用可能かを確認
    # print(torch.__version__)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, default="/root/save")
    parser.add_argument('--csv_path', type=str, default='/root/save/dataset/Task01_BrainTumour/id_list.csv')

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=3)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=3)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='MDM')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='MDM_2D')
    parser.add_argument('--dataset', type=str, default='WDDD2_2D')
    parser.add_argument('--model_detail', type=str, default='masked diffusionで特徴を学習する')   # ちょっとした詳細

    parser.add_argument('--train_size', type=int, default=0)  # 訓練
    parser.add_argument('--val_size', type=int, default=0)    # 検証
    parser.add_argument('--test_size', type=int, default=0)   # テスト

    parser.add_argument('--img_size', type=str, default='128×128×64')       # 画像サイズ (wandbに送るよう)
    parser.add_argument('--optimizer', type=str, default='Adam')         # 最適化関数 (wandbに送るよう)
    parser.add_argument('--loss_function', type=str, default='SSIM')  # 損失関数 (wandbに送るよう)
    parser.add_argument('--wandb_num_images', type=int, default=4)
    parser.add_argument('--save_n_model', type=int, default=50)
    parser.add_argument('--val_volume', type=int, default=0)
    parser.add_argument('--train_volume', type=int, default=0)

    parser.add_argument('--dim_size', type=str, default='2D')       # 2D or 3D (wandbに送るよう)
    args = parser.parse_args()
    main(args)