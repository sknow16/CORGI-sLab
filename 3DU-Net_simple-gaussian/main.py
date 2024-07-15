from dataset import brain_dataset
from model import UNet3D
import torch
import pandas as pd

from monai.transforms import (
    Compose,
    Resized,
    ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesD,
)
from tqdm.auto import tqdm
import argparse
from sklearn.model_selection import train_test_split
from train_util import Trainer
from torch.optim import Adam

from guided_diffusion.script_util import create_gaussian_diffusion

def main(args):
    model     = UNet3D(args.in_channels, args.out_channels)
    diffusion = create_gaussian_diffusion(
        # 拡散処理の設定
        steps=1000,         # 時間ステップ:T
        learn_sigma=False,  # 分散を学習するか
        sigma_small=False,
        noise_schedule="linear",  # ノイズのスケジュール
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="ddim50", # 何も指定しなければddpm, ddim100
        # timestep_respacing="ddim100", # 何も指定しなければddpm, ddim100
    )

    df = pd.read_csv(args.csv_path)
    # 画像の前処理 (今は、ラベルの情報が0,1,2,3になっていたのがResizeでおかしくなってることに注意)
    transform = Compose([
        ConvertToMultiChannelBasedOnBratsClassesD(keys="label"),
        Resized(keys=["image"], spatial_size=(args.image_size, args.image_size, args.volume_size), mode="trilinear"),  # 画像のサイズ変更
        Resized(keys=["label"], spatial_size=(args.image_size, args.image_size, args.volume_size), mode="nearest"),  # ラベルのサイズ変更
        ToTensord(keys=["image", "label"]),
    ])

    val_test_df, train_df = train_test_split(df, test_size=0.515, random_state=args.seed)
    val_test_df, rest_df = train_test_split(val_test_df, test_size=0.17, random_state=args.seed)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=args.seed)
    args.train_size, args.val_size, args.test_size = len(train_df), len(val_df), len(test_df)
    
    train_set = brain_dataset(train_df, transform=transform)
    val_set   = brain_dataset(val_df, transform=transform)
    test_set  = brain_dataset(test_df, transform=transform)
    
    optimizer = Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.MSELoss()
    trainer = Trainer(
        model=model,
        diffusion = diffusion,
        optimizer=optimizer,
        criterion=criterion,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        args=args,
        dir_path=None,
    )
    
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csv_path', type=str, default='/root/save/dataset/Task01_BrainTumour/id_list.csv')

    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=3)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=3)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='3DU-Net_Brats_128zmini')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='3DU-Net_gaussian')
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