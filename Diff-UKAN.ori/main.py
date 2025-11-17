from datetime import date
import os
from Brats2021 import preDataset
from model import DiffUNet
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
from save_log import load_model, create_folder
from guided_diffusion.script_util import create_gaussian_diffusion

import torch


def estimate_memory_usage(model, diffusion, dataset, device="cuda"):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    model = model.to(device)
    model.train()

    batch = next(iter(dataloader))
    image = batch['image'].to(device)
    mask = batch['label'].to(device)

    torch.cuda.reset_peak_memory_stats(device)

    # forward
    t = torch.randint(0, diffusion.num_timesteps, (mask.shape[0],), device=device)
    model_kwargs = dict(image=image)
    loss_dict = diffusion.training_losses(model, mask, t, model_kwargs)
    loss = loss_dict["loss"]

    # backward
    loss.backward()

    current = torch.cuda.memory_allocated(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"\n[ VRAM USAGE ESTIMATION ]")
    print(f"Current Memory   : {current:.2f} MB")
    print(f"Peak Memory Used : {peak:.2f} MB (forward + backward)")
    # GB表記
    print(f"Current Memory   : {current/1024:.2f} GB")
    print(f"Peak Memory Used : {peak/1024:.2f} GB (forward + backward)")
    

    # cleanup
    del batch, image, mask, t, model_kwargs, loss_dict, loss, dataloader
    torch.cuda.empty_cache()



def main(args):
    date_str = date.today().strftime("%Y%m%d")
    dir_path = os.path.join(args.path,"log",date_str+":"+args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
    create_folder(dir_path)
    model     = DiffUNet()
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

    # 画像の前処理 (今は、ラベルの情報が0,1,2,3になっていたのがResizeでおかしくなってることに注意)
    transform = Compose([
        ConvertToMultiChannelBasedOnBratsClassesD(keys="label"),
        Resized(keys=["image"], spatial_size=(args.image_size, args.image_size, args.volume_size), mode="trilinear"),  # 画像のサイズ変更
        Resized(keys=["label"], spatial_size=(args.image_size, args.image_size, args.volume_size), mode="nearest"),  # ラベルのサイズ変更
        ToTensord(keys=["image", "label"]),
    ])

    data_list = [os.path.join(args.dataset_path, d) for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    train_list , test_list = train_test_split(data_list, test_size=0.2, random_state=args.seed)
    test_list , val_list  = train_test_split(test_list,  test_size=0.5, random_state=args.seed)

    args.train_size, args.val_size, args.test_size = len(train_list), len(val_list), len(test_list)
    train_set = preDataset(train_list, transform=transform)
    val_set   = preDataset(val_list,   transform=transform)
    test_set  = preDataset(test_list,  transform=transform)
    
    optimizer = Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.MSELoss()
    # モデルのパラメータ数を計算
    num_params = sum(p.numel() for p in model.parameters())
    print(f"モデルのパラメータ数: {num_params}")
    # Milion表記
    print(f"モデルのパラメータ数: {num_params/1e6:.2f} Million")
    # 学習可能なパラメータ数を計算
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"学習可能なパラメータ数: {num_trainable_params}")
    print(f"学習可能なパラメータ数: {num_trainable_params/1e6:.2f} Million")
    
    # 学習で必要な最大メモリを計測
    estimate_memory_usage(model, diffusion, train_set, device="cuda")
    
    trainer = Trainer(
        model=model,
        diffusion = diffusion,
        optimizer=optimizer,
        criterion=criterion,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        args=args,
        dir_path=dir_path,
        num_params=num_params,
        num_trainable_params=num_trainable_params,    
    )
    
    trainer.train()  # 学習実行するコード
    trainer.test(args) # テスト実行するコード

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=40) # 今まで42
    parser.add_argument('--path', type=str, default="/root/save")
    parser.add_argument('--dataset_path', type=str, default="/root/save/dataset/Brats2021/BraTS2021_Training_Data")

    parser.add_argument('--image_size', type=int, default=240)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=3)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=3)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='Brats2021_24024064')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='Diff-UNet.ref')
    parser.add_argument('--dataset', type=str, default='Brats2021')
    parser.add_argument('--model_detail', type=str, default='Diff-UNetをまねした')   # ちょっとした詳細

    parser.add_argument('--train_size', type=int, default=0)  # 訓練
    parser.add_argument('--val_size', type=int, default=0)    # 検証
    parser.add_argument('--test_size', type=int, default=0)   # テスト

    parser.add_argument('--img_size', type=str, default='240×240×64')       # 画像サイズ (wandbに送るよう)
    parser.add_argument('--optimizer', type=str, default='Adam')         # 最適化関数 (wandbに送るよう)
    parser.add_argument('--loss_function', type=str, default='MSELoss')  # 損失関数 (wandbに送るよう)

    parser.add_argument('--dim_size', type=str, default='3D')       # 2D or 3D (wandbに送るよう)
    args = parser.parse_args()
    main(args)