from Part1_position_model.patchsize_128_2D import UNet
from Part2_segmentation_model.patchsize_128_2D import PatchSegUnet
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
import argparse
from sklearn.model_selection import train_test_split
from train_util import Trainer
from torch.optim import Adam
from guided_diffusion.save_log import create_folder
import os
from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.save_log import load_model
def main(args):

    transform = {
        "image": transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),  
            transforms.RandomHorizontalFlip(p=0.5), # y軸方向を中心に反転
            transforms.RandomVerticalFlip(p=0.5),  # x軸方向を中心に反転
            transforms.ToTensor()  # Tensorに変換
            ]),
        "mask": transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.NEAREST),  
            transforms.RandomHorizontalFlip(p=0.5), # y軸方向を中心に反転
            transforms.RandomVerticalFlip(p=0.5),  # x軸方向を中心に反転
            ]),
    }

    val_transform = {
        "image": transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR), 
            transforms.ToTensor()  # Tensorに変換
            ]),
        "mask": transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.NEAREST),  
            ]),
    }

    if args.task_select == 'part1':        
        from wddd2_dataset_pos import WDDD2Dataset
        train_set   = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftrain.csv', transform=transform)
        test_set    = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftest.csv', transform=val_transform)
        val_set     = test_set
        
        model = UNet(in_ch=args.in_channels, out_ch=args.out_channels, cond_in_ch=2 ,hidden_ch=32, time_embed_dim=100)
        pos_encoder = None

    elif args.task_select == 'part2':
        from wddd2_dataset_Seg import SegWDDD2Dataset
        train_set = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segtrain.csv', transform=transform)
        val_set   = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segval.csv', transform=val_transform)
        test_set  = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segtest.csv', transform=val_transform)

        part1_model = UNet(in_ch=1, out_ch=1, cond_in_ch=2 ,hidden_ch=32, time_embed_dim=100)
        part1_model = load_model(part1_model, args.part1_path)
        pos_encoder = part1_model.position_encoder
        model = PatchSegUnet(in_ch=args.in_channels, out_ch=args.out_channels, cond_in_ch=1 ,hidden_ch=32, time_embed_dim=100)
        
    diffusion = create_gaussian_diffusion(
        # 拡散処理の設定
        steps=1000,         # 時間ステップ:T
        learn_sigma=False,  # 分散を学習するか
        sigma_small=False,
        noise_schedule="linear",  # ノイズのスケジュール
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="ddim50", # 何も指定しなければddpm, ddim100
        # timestep_respacing="ddim100", # 何も指定しなければddpm, ddim100
    )

    dir_path = os.path.join(args.path,"log",args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
    create_folder(dir_path)



    optimizer = Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.MSELoss()
    trainer = Trainer(
        model=model,
        pos_encoder = pos_encoder,
        diffusion = diffusion,
        optimizer=optimizer,
        criterion=criterion,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        args=args,
        dir_path=dir_path,
    )
    
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_select', type=str, default='part2')  # pasrt1 or part2
    parser.add_argument('--part1_path', type=str, default="/root/save/log/masked_gen_128_epochs_200:WDDD2/weights/weight_epoch_best.pth") # part1 modelの学習済みの重みパス
    parser.add_argument('--check_point_path', type=str, default="/root/save/log/PatchSegUnet128_notdonut_epochs_1000:WDDD2/weights/weight_epoch_last.pth") # part2 セグメンテーションの追加学習したいときのパス
    parser.add_argument('--check_point', type=bool, default=True) # True 追加学習するとき
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, default="/root/save")

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=3)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=3)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--crop_size', type=int, default=128)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='Patch_Seg_2D')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='PatchSegUnet128_notdonut')
    parser.add_argument('--dataset', type=str, default='WDDD2')
    parser.add_argument('--model_detail', type=str, default='正解データのチャネルの値を変えた')   # ちょっとした詳細

    parser.add_argument('--train_size', type=int, default=0)  # 訓練
    parser.add_argument('--val_size', type=int, default=0)    # 検証
    parser.add_argument('--test_size', type=int, default=0)   # テスト

    parser.add_argument('--img_size', type=str, default='256×256')       # 画像サイズ (wandbに送るよう)
    parser.add_argument('--optimizer', type=str, default='Adam')         # 最適化関数 (wandbに送るよう)
    parser.add_argument('--loss_function', type=str, default='MSELoss')  # 損失関数 (wandbに送るよう)

    parser.add_argument('--dim_size', type=str, default='2D')       # 2D or 3D (wandbに送るよう)
    args = parser.parse_args()
    main(args)