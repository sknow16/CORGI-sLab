# 比較用の2DU-Net
# 実行時のみフォルダからだしてmain.pyと同じ階層で実行

import torch.nn as nn
import torch
import math
from tqdm.auto import tqdm
from torchvision import transforms

from torchvision.transforms import InterpolationMode
import argparse

from torch.utils.data import DataLoader
from wddd2_dataset_Seg import SegWDDD2Dataset
from skimage.filters import threshold_otsu
from metric.metric import jaccard   
import wandb
from guided_diffusion.save_log import save_model
from guided_diffusion.save_log import create_folder
import os

def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch,dropout=0, useconv=False):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        
        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        elif useconv:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_ch, out_ch, 3, padding=1)
            ),
        )

    def forward(self, x):
        h = self.in_layer(x)
        h = self.out_layer(h)
        return h + self.skip_connection(x)
        # if v is not None:
        #     emb_out = self.time_emb(v)
        #     while len(emb_out.shape) < len(h.shape):
        #         emb_out = emb_out[..., None]
            
        #     out_norm, out_rest = self.out_layer[0], self.out_layer[1:]
        #     shift, scale = emb_out.chunk(2, dim=1)
        #     h = out_norm(h) * (1 + scale) + shift
        #     h = out_rest(h)
        # else:
        #     h = self.out_layer(h)
        # return h+self.skip_connection(x)

# U-Net構造 ============================================================================================
# 2D用
# シンプルなモデルにResBlockの層を追加したもの
# ======================================================================================================
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, hidden_ch=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3,1,1)  # パッチ画像(1, 64, 64) -> (32, 64, 64)に変更
        self.down1 = ResBlock(hidden_ch, hidden_ch*2)
        self.down2 = ResBlock(hidden_ch*2, hidden_ch*4)
        self.down3 = ResBlock(hidden_ch*4, hidden_ch*8)

        self.bot1  = ResBlock(hidden_ch*8, hidden_ch*8)

        self.up3 = ResBlock(hidden_ch*8+hidden_ch*8, hidden_ch*4)
        self.up2 = ResBlock(hidden_ch*4+hidden_ch*4, hidden_ch*2)
        self.up1 = ResBlock(hidden_ch*2+hidden_ch*2, hidden_ch)
        self.out = nn.Conv2d(hidden_ch, out_ch, 1)
        
        self.maxpool  = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv_in(x)
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        x3 = self.down3(x)
        x = self.maxpool(x3)
        x = self.bot1(x)

        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.out(x)
        return x


def main(args):
    dir_path = os.path.join(args.path,"log",args.model_name+"_epochs_"+str(args.epochs)+":"+args.dataset)
    create_folder(dir_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(args.in_channels, args.out_channels)
    model = model.to(device)

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
    train_set = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segtrain.csv', transform=transform)
    val_set   = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segval.csv', transform=val_transform)
    test_set  = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segtest.csv', transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=False)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)


    best_iou = 0

    if args.wandb_flag:
        wandb.init(
                name=args.model_name,
                project=args.project_name,
                tags=[args.dim_size],
                notes=args.model_detail,
                config={
                "model":         args.model_name,
                "dataset":       args.dataset,
                "epochs":        args.epochs,
                "image_size":    args.img_size,
                "in_channel":    args.in_channels,
                "out_channel":   args.out_channels,

                "crop_size":     args.crop_size,
                "batch_size":    args.batch_size,
                "learning_rate": args.lr,

                "optimizer": args.optimizer,         # 使用している最適化手法
                "loss_function": args.loss_function, # 使用している損失関数

                "train_volume": args.train_size,
                "val_volume": args.val_size,
                "test_volume": args.test_size
                }
        )
            


    for epoch in range(args.epochs):
        train_loss_list  = []
        val_ch0_iou_list = []
        val_ch1_iou_list = []
        val_ch2_iou_list = []
        print(f"{epoch+1}/{args.epochs}")
        model.train()
        for batch in tqdm(train_loader):
            img, mask = batch
            img = img.to(device)
            mask = mask.to(device)
            pred_mask = model(img)
            loss = nn.MSELoss()(pred_mask, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_list.append(loss.item())
        
        model.eval()    
        for batch in tqdm(val_loader):
            img, mask = batch
            img  = img.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)

            threshold_ch0 = threshold_otsu(pred_mask[:,0].cpu().numpy())
            threshold_ch1 = threshold_otsu(pred_mask[:,1].cpu().numpy())
            threshold_ch2 = threshold_otsu(pred_mask[:,2].cpu().numpy())

            # 各チャネルを二値化
            pred_binary_ch0 = (pred_mask[:,0] > threshold_ch0).float()
            pred_binary_ch1 = (pred_mask[:,1] > threshold_ch1).float()
            pred_binary_ch2 = (pred_mask[:,2] > threshold_ch2).float()

            # 3チャネルに結合
            pred_binary_mask = torch.stack([pred_binary_ch0, pred_binary_ch1, pred_binary_ch2], dim=1)  # [N, 3, H, W]
            mask = mask.cpu().numpy()
                        
            for i in range(mask.shape[0]):
                val_ch0_iou_list.append(jaccard(pred_binary_ch0[i].cpu().numpy(), mask[i][0]))
                val_ch1_iou_list.append(jaccard(pred_binary_ch1[i].cpu().numpy(), mask[i][1]))
                val_ch2_iou_list.append(jaccard(pred_binary_ch2[i].cpu().numpy(), mask[i][2]))

        train_average_loss  = sum(train_loss_list)/len(train_loss_list)
        val_ch0_iou_averege = sum(val_ch0_iou_list)/len(val_ch0_iou_list) # 背景
        val_ch1_iou_averege = sum(val_ch1_iou_list)/len(val_ch1_iou_list) # 核
        val_ch2_iou_averege = sum(val_ch2_iou_list)/len(val_ch2_iou_list) # 胚
        val_all_iou_average = (val_ch1_iou_averege+val_ch2_iou_averege)/2 # 核と胚の平均
        if best_iou < val_all_iou_average:
            print(f"update best score: {val_all_iou_average}")
            save_model(model, 'best', dir_path)
    
    
        wandb_image=[wandb.Image(img[i]) for i in range(len(img))]
        wandb_mask =[wandb.Image(mask[i].transpose(1, 2, 0)) for i in range(len(mask))]
        wandb_pred_mask    =[wandb.Image(pred_binary_mask[i]) for i in range(len(pred_binary_mask))]  

        wandb.log({
                'train_loss':train_average_loss,
                                '背景IoU': val_ch0_iou_averege,
                                '核IoU': val_ch1_iou_averege,
                                '胚IoU': val_ch2_iou_averege,
                                '胚と核の平均IoU': val_all_iou_average,

                                'image':wandb_image,
                                'mask':wandb_mask,
                                # 'patch_image': wandb_patch_image,
                                # 'x':wandb_x,
                                'pred_mask':wandb_pred_mask,
                                # '背景の予測':pred_binary_ch0,
                                # '核の予測':pred_binary_ch1,
                                # '胚の予測':pred_binary_ch2,
                                # 'cond':wandb_cond,
            })

    save_model(model, 'last', dir_path)
    
    if True:
        test_ch0_iou_list = []
        test_ch1_iou_list = []
        test_ch2_iou_list = []

        model.eval()    
        for batch in tqdm(test_loader):
            img, mask = batch
            img  = img.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)

            threshold_ch0 = threshold_otsu(pred_mask[:,0].cpu().numpy())
            threshold_ch1 = threshold_otsu(pred_mask[:,1].cpu().numpy())
            threshold_ch2 = threshold_otsu(pred_mask[:,2].cpu().numpy())

            # 各チャネルを二値化
            pred_binary_ch0 = (pred_mask[:,0] > threshold_ch0).float()
            pred_binary_ch1 = (pred_mask[:,1] > threshold_ch1).float()
            pred_binary_ch2 = (pred_mask[:,2] > threshold_ch2).float()

            # 3チャネルに結合
            pred_binary_mask = torch.stack([pred_binary_ch0, pred_binary_ch1, pred_binary_ch2], dim=1)  # [N, 3, H, W]
            mask = mask.cpu().numpy()
                        
            for i in range(mask.shape[0]):
                test_ch0_iou_list.append(jaccard(pred_binary_ch0[i].cpu().numpy(), mask[i][0]))
                test_ch1_iou_list.append(jaccard(pred_binary_ch1[i].cpu().numpy(), mask[i][1]))
                test_ch2_iou_list.append(jaccard(pred_binary_ch2[i].cpu().numpy(), mask[i][2]))

        # train_average_loss  = sum(train_loss_list)/len(train_loss_list)
        test_ch0_iou_averege = sum(test_ch0_iou_list)/len(test_ch0_iou_list) # 背景
        test_ch1_iou_averege = sum(test_ch1_iou_list)/len(test_ch1_iou_list) # 核
        test_ch2_iou_averege = sum(test_ch2_iou_list)/len(test_ch2_iou_list) # 胚
        test_all_iou_average = (test_ch1_iou_averege+test_ch2_iou_averege)/2 # 核と胚の平均
        wandb_image=[wandb.Image(img[i]) for i in range(len(img))]
        wandb_mask =[wandb.Image(mask[i].transpose(1, 2, 0)) for i in range(len(mask))]
        wandb_pred_mask    =[wandb.Image(pred_binary_mask[i]) for i in range(len(pred_binary_mask))]  

        wandb.log({
                                '背景IoU': test_ch0_iou_averege,
                                '核IoU': test_ch1_iou_averege,
                                '胚IoU': test_ch2_iou_averege,
                                '胚と核の平均IoU': test_all_iou_average,

                                'image':wandb_image,
                                'mask':wandb_mask,

                                'pred_mask':wandb_pred_mask,
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--part1_path', type=str, default="/root/save/log/masked_gen_128_epochs_200:WDDD2/weights/weight_epoch_best.pth") # part1 modelの学習済みの重みパス
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, default="/root/save")

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--volume_size', type=int, default=64)

    parser.add_argument('--in_channels', type=int, default=1)   # モデルの入力チャネル
    parser.add_argument('--out_channels', type=int, default=3)  # モデルの出力チャネル

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--crop_size', type=int, default=128)

    # wandb関連
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='Patch_Seg_2D')         # プロジェクト名
    parser.add_argument('--model_name', type=str, default='other_model(U-Net)_notDonut')
    parser.add_argument('--dataset', type=str, default='WDDD2')
    parser.add_argument('--model_detail', type=str, default='比較用モデル(2DU-Net)')   # ちょっとした詳細

    parser.add_argument('--train_size', type=int, default=0)  # 訓練
    parser.add_argument('--val_size', type=int, default=0)    # 検証
    parser.add_argument('--test_size', type=int, default=0)   # テスト

    parser.add_argument('--img_size', type=str, default='256×256')       # 画像サイズ (wandbに送るよう)
    parser.add_argument('--optimizer', type=str, default='Adam')         # 最適化関数 (wandbに送るよう)
    parser.add_argument('--loss_function', type=str, default='MSELoss')  # 損失関数 (wandbに送るよう)

    parser.add_argument('--dim_size', type=str, default='2D')       # 2D or 3D (wandbに送るよう)
    args = parser.parse_args()
    main(args)