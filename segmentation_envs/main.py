from Part1_position_model.default_2D import UNet
# from Part2_segmentation_model.default_2D import SegUnet
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  
        transforms.RandomHorizontalFlip(p=0.5), # y軸方向を中心に反転
        transforms.RandomVerticalFlip(p=0.5),  # x軸方向を中心に反転
        transforms.ToTensor()  # Tensorに変換
    ])
    if args.task_select == 'part1':
        model = UNet(in_ch=args.in_channels, out_ch=args.out_channels, cond_in_ch=1 ,hidden_ch=32, time_embed_dim=100)
        
        from wddd2_dataset_pos import WDDD2Dataset
        Ftraindataset   = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftrain.csv', transform=transform)
        Ftestdataset    = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftest.csv', transform=transform)

    elif args.task_select == 'part2':
        part1_model = UNet(in_ch=args.in_channels, out_ch=args.out_channels, cond_in_ch=1 ,hidden_ch=32, time_embed_dim=100)
        part1_model = load_model(part1_model, args.part1_path)
        pos_enc = part1_model.position_encoder
        model = SegUnet(in_ch=args.in_channels, out_ch=args.out_channels, cond_in_ch=1 ,hidden_ch=32, time_embed_dim=100)
        
        from wddd2_dataset import WDDD2Dataset

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
        diffusion = diffusion,
        optimizer=optimizer,
        criterion=criterion,
        train_set=Ftraindataset,
        val_set=Ftestdataset,
        test_set=Ftestdataset,
        args=args,
        dir_path=dir_path,
    )
    
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_select', type=str, default='part1')  # part1 or part2
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, default="/root/save")

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=1)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=1)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='crop_wddd2')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='masked_gen')
    parser.add_argument('--dataset', type=str, default='WDDD2')
    parser.add_argument('--model_detail', type=str, default='2D画像の一部をマスクして再構成する, クロップ部を0ではなく輝度を変更(+0.3)')   # ちょっとした詳細

    parser.add_argument('--train_size', type=int, default=0)  # 訓練
    parser.add_argument('--val_size', type=int, default=0)    # 検証
    parser.add_argument('--test_size', type=int, default=0)   # テスト

    parser.add_argument('--img_size', type=str, default='256×256')       # 画像サイズ (wandbに送るよう)
    parser.add_argument('--optimizer', type=str, default='Adam')         # 最適化関数 (wandbに送るよう)
    parser.add_argument('--loss_function', type=str, default='MSELoss')  # 損失関数 (wandbに送るよう)

    parser.add_argument('--dim_size', type=str, default='2D')       # 2D or 3D (wandbに送るよう)
    args = parser.parse_args()
    main(args)