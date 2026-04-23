from collections import defaultdict

from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import wandb
from skimage.filters import threshold_otsu
from metric.metric import jaccard

import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
# from pytorch_msssim import ssim
import torchgeometry as tgm
from utils.utils import load_model, save_model
import os
from utils import requires_grad, update_ema, calc_metric
from dataset.wddd2.wddd2_dataset_Seg import fundus_inv_map_mask
# x = torch.randn(1,1,32,32)
# mask = torch.zeros(8,8)

def random_coordinate(mask_point_range, batch):
    coordinates = [(random.randint(0, mask_point_range), random.randint(0, mask_point_range)) for _ in range(batch)]
    return coordinates

crop_color = 0.3
def create_crop(x, crop_size, mask=None):
    n, c, h, w = x.shape
    left_points = random_coordinate(w-crop_size, n)
    mini_x = torch.zeros(n, c, crop_size, crop_size)
    x_mask_ch = torch.zeros(n, c, h, w)
    if mask is not None:
        n, c, h, w = mask.shape
        mini_mask = torch.zeros(n, c, crop_size, crop_size)
    for i in range(n):
        mini_x[i] = x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]
        if mask is not None:
            mini_mask[i] = mask[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]
        # x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size] = 0
        x_mask_ch[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size] = torch.clamp(x_mask_ch[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]+1, 0, 1)
    new_x = torch.cat((x, x_mask_ch), dim=1)
    if mask is not None:
        return mini_x, new_x, mini_mask
    else:
        return mini_x, new_x 

def inference_image_split(image, crop_size):
    coordinates = []
    _, _, height, width = image.shape
    for y in range(height-crop_size, -1, -crop_size):
        for x in range(width-crop_size, -1, -crop_size):
            coordinates.append((x, y))
    return coordinates

# オーバーラップの座標を出す
def overlap_image_split(image, crop_size):
    overlap_coordinates = []
    _, _, height, width = image.shape
    crop_size_half = crop_size/2
    crop_size_half = int(crop_size_half)
    
    pair0 = 0
    pair1 = crop_size_half
    while pair1 < height-crop_size_half:
        print(f"pair0: {pair0}, pair1: {pair1}")
        overlap_coordinates.append((pair0, pair1))
        overlap_coordinates.append((pair1, pair0))
        pair0 += crop_size_half
        pair1 += crop_size_half
    return overlap_coordinates

def create_inference_crop(x, crop_size, mask, left_point):
    n, c, h, w = x.shape
    mini_x = torch.zeros(n, c, crop_size, crop_size)
    x_mask_ch = torch.zeros(n, c, h, w)
    if mask is not None:
        n, c, h, w = mask.shape
        mini_mask = torch.zeros(n, c, crop_size, crop_size)
    mini_x    = x[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]
    mini_mask = mask[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]
    # x[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size] = torch.clamp(x[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]+crop_color, 0, 1)
    x_mask_ch[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size] = torch.clamp(x_mask_ch[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]+1, 0, 1)
    new_x = torch.cat((x, x_mask_ch), dim=1)
    if mask is not None:
        return mini_x, new_x, mini_mask
    else:
        return mini_x, new_x, 

    
# mini_x, x = create_crop(x, 4)

class Trainer:
    def __init__(
            self,
            *, # *以降は呼び出す際にキーワード引数で指定する必要がある
            model,
            diffuser,
            optimizer,
            train_set,
            val_set,
            test_set,
            cfg,
            dir_path,
            vae=None,
            pos_encoder=None,
            criterion=None,
            scheduler=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["checkpoint"]["use_checkpoint"]:
            self.checkpoint_epoch = cfg["checkpoint"]["checkpoint_epoch"]
            model = load_model(model, cfg["checkpoint"]["path"])
            print("check pointから追加学習を開始します")
        else:
            self.checkpoint_epoch = 0
            
        self.model  = model.to(self.device)
        if cfg["train"]["use_ema"]:
            print("Using EMA Model...")
            self.ema = deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
            requires_grad(self.ema, False)
        if pos_encoder is not None:
            self.pos_encoder = pos_encoder.to(self.device)
            self.pos_encoder.eval()
        if vae is not None:
            self.vae = vae.to(self.device)
            self.vae.eval()
            self.scaling_factor = cfg["vae"]["scaling_factor"]
        
        self.task_select    = cfg["task"]
        self.diffuser_type  = cfg["diffuser_type"]
        self.diffuser       = diffuser
        self.optimizer      = optimizer
        self.criterion      = criterion # 拡散モデル以外で使用

        self.train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size(global)"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(val_set, batch_size=cfg["train"]["batch_size(global)"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=False)
        self.test_loader  = DataLoader(test_set, batch_size=cfg["train"]["batch_size(global)"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=False)

        self.space       = cfg["space"]
        self.crop_size   = cfg["crop_size"]       # null or 64 or 128
        self.train_size  = cfg["data"]["train_size"]
        self.test_size   = cfg["data"]["test_size"]
        self.val_size    = cfg["data"]["val_size"]
        self.epochs      = cfg["train"]["epochs"]
        self.clip_grad   = cfg["train"]["clip_grad"]
        self.rank        = 0 # 現在は単一想定
        self.microbatch  = cfg["train"]["batch_size(micro)"]
        self.use_ema     = cfg["train"]["use_ema"]
        
        self.dir_path = dir_path
        if self.task_select=="pos":
            self.best_score = float('inf')
        elif self.task_select=="seg":
            self.best_score = 0 # mIoUが高いほど良いので初期値は0
        
        self.best_path            = os.path.join(dir_path,"weights",f"weight_epoch_best_dice.pth")
        self.ema_best_path        = os.path.join(dir_path,"weights",f"weight_epoch_best_dice_ema.pth")
        self.last_path            = os.path.join(dir_path,"weights",f"weight_epoch_last.pth")
        self.ema_last_path        = os.path.join(dir_path,"weights",f"weight_epoch_last_ema.pth")

        self.image_shape  = (cfg["data"]["img_channels"], cfg["data"]["img_size"], cfg["data"]["img_size"])
        self.mask_shape   = (cfg["data"]["mask_channels"], cfg["data"]["img_size"], cfg["data"]["img_size"])
        if self.space == "latent":
            self.latent_shape = (cfg["vae"]["latent"]["latent_dim"], cfg["vae"]["latent"]["latent_size"], cfg["vae"]["latent"]["latent_size"])
        
    def train_loop(self,cfg):
        tags = [cfg["model"]["model_name"], cfg["data"]["dataset"], cfg["diffuser_type"], cfg["space"]]
        tags = [t for t in tags if t is not None]
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["run_name"],
            tags=tags,
            config=cfg,
            save_code=cfg["wandb"]["save_code"],
        )
        if cfg["train"]["use_ema"]:
            if cfg["checkpoint"]["use_checkpoint"]:
                print("Reading the EMA checkpoint...")
                self.ema = load_model(self.ema, cfg["checkpoint"]["ema_path"]).to(self.device)
                requires_grad(self.ema, False)
            else:
                update_ema(self.ema, self.model, decay=0)  # EMAモデルの初期化
        
        for epoch in range(self.checkpoint_epoch+1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            train_losses = []
            
            for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=self.rank != 0):
                self.model.train()
                image, mask = batch
                self.optimizer.zero_grad()
                for i in range(0, mask.shape[0],self.microbatch):
                    micro_mask   = mask[i:i+self.microbatch]
                    micro_image  = image[i:i+self.microbatch]
                    micro_mask   = micro_mask.to(self.device)
                    x_start      = micro_mask.to(self.device)  # セグメンテーションマップ
                    y            = micro_image.to(self.device) # 条件画像
                    loss = self._forward(x_start, y, cfg)
                    loss.backward()
                    train_losses.append(loss.item() * len(x_start))
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                if self.use_ema:
                    update_ema(self.ema, self.model)  # EMAモデルの更新
            train_avg_loss = sum(train_losses) / self.train_size
            wandb_lr = self.optimizer.param_groups[0]['lr']
            if epoch % cfg["train"]["val_step_num"] == 0:
                log_config = self.val_loop(cfg)
            else:
                log_config = {}
            log_config.update({
                "train_loss": train_avg_loss,
                "lr": wandb_lr,
            })
            
            wandb.log(log_config)
            # 5回に1回モデルを保存(保存時は一個前の保存を削除)
            if epoch % cfg["train"]["save_n_model"] == 0 and epoch != self.epochs:
                save_model(
                    self.model, epoch, self.dir_path,
                    optimizer = None,
                    ema       = self.ema if self.use_ema else None,
                )

        # 最後のモデルを保存
        save_model(
            self.model, 'last', self.dir_path,
            optimizer = None,
            ema       = self.ema if self.use_ema else None,
        )
        wandb.finish()

    def val_loop(self, cfg):
        all_metrics = defaultdict(list)
        
        if self.use_ema:
            self.ema.eval()
        self.model.eval()
        for image, mask in tqdm(self.val_loader, desc="Validation", disable=self.rank != 0):
            x_start = mask.to(self.device)
            y       = image.to(self.device)
            pred_x_start_list = []
            if self.diffuser_type is None:
                pred_mask = self._forward_inference(x_start, y, cfg)
                std_mask  = None
            else:
                for _ in range(cfg["diffusion"]["val_ensemble"]):
                    pred_x_start = self._forward_inference(x_start, y, cfg)
                    pred_x_start_list.append(pred_x_start)
                # 平均, 標準偏差を計算
                pred_mask   = torch.mean(torch.stack(pred_x_start_list), dim=0) # emsembleの平均
                std_mask    = torch.std(torch.stack(pred_x_start_list), dim=0)  # emsembleの標準偏差
            pred_binary_mask, metrics = self._calc_metric_binary(pred_mask, mask)
            
            for ch, metric in metrics.items():
                for key, value in metric.items():
                    all_metrics[f"{ch}_{key}"].append(value)
        # ループ後の平均
        avg_metrics = {k: sum(v) / self.val_size for k, v in all_metrics.items()}
        #avg_metricsにmiouも追加
        avg_metrics["miou"] = (avg_metrics["ch1_iou"] + avg_metrics["ch2_iou"]) / 2
        # print(f"Validation Metrics: {avg_metrics}")
        # 最後のバッチだけwandbへ送る
        # mask = fundus_inv_map_mask(mask.cpu().numpy())
        # pred_binary_mask = fundus_inv_map_mask(pred_binary_mask.cpu().numpy())
        y = y.cpu().numpy()
        mask = mask.cpu().numpy()
        pred_binary_mask = pred_binary_mask.cpu().numpy()
        std_mask = std_mask.cpu().numpy() if std_mask is not None else None
        
        wandb_num_images  = min(cfg["wandb"]["send_images"], y.shape[0], x_start.shape[0])
        wandb_y           = [wandb.Image(y[i].transpose(1, 2, 0)) for i in range(wandb_num_images)]
        wandb_x           = [wandb.Image(mask[i].transpose(1, 2, 0)) for i in range(wandb_num_images)]
        pred_binary_x     = [wandb.Image(pred_binary_mask[i].transpose(1, 2, 0)) for i in range(wandb_num_images)]
        
        log_dict = {
            "x": wandb_x,
            "y": wandb_y,
            "pred_binary_x": pred_binary_x,
        }
        # stdは拡散モデルのときだけ追加
        if std_mask is not None:
            log_dict["std"] = [wandb.Image(std_mask[i].transpose(1, 2, 0)) for i in range(wandb_num_images)]
        log_dict.update(avg_metrics)
        
        if self.best_score < (avg_metrics["ch1_iou"]+avg_metrics["ch2_iou"])/2: # チャンネル1とチャンネル2のIoUの平均が高いほど良いとする
            print(f"ベストmIoU更新: {self.best_score:.4f} → {(avg_metrics['ch1_iou']+avg_metrics['ch2_iou'])/2:.4f}")
            self.best_score = (avg_metrics["ch1_iou"]+avg_metrics["ch2_iou"])/2 
            save_model(self.model, 'best', self.dir_path)
            if self.use_ema:
                save_model(self.ema, 'best_ema', self.dir_path)
                
        return log_dict
      
    def _build_model_kwargs(self, y, cfg): # モデルへのx以外の入力をどうするか
        model_name = cfg["model"]["model_name"]
        if model_name == "LSegDiff":
            model_kwargs = dict(conditioned_image=y)
        else:
            model_kwargs = dict()
        return model_kwargs
    
    def _forward(self, x_start, y, cfg):
        model_kwargs = self._build_model_kwargs(y, cfg)
        if self.space == "latent":
            with torch.no_grad():
                x_start = self.vae.encode(x=x_start).latent_dist
                if cfg["vae"]["latent"]["latent_sample"] == "mean":
                    x_start = x_start.mode()
                elif cfg["vae"]["latent"]["latent_sample"] == "sample":
                    x_start = x_start.sample()
                x_start = x_start.mul_(self.scaling_factor)
        if self.diffuser_type == "diffusion":
            t = torch.randint(0, self.diffuser.num_timesteps, (x_start.shape[0],), device=self.device)
            loss_dict = self.diffuser.training_losses(self.model, x_start, t, model_kwargs)
            loss      = loss_dict["loss"]
        elif self.diffuser_type == "rectified_flow":
            noise = torch.randn_like(x_start)
            loss = self.diffuser.train_losses(self.model, z0=noise, z1=x_start,model_kwargs=model_kwargs)
        elif self.diffuser_type is None:
            pred_x = self.model(y, **model_kwargs)
            loss = self.criterion(pred_x, x_start)
        return loss
    
    def _forward_inference(self, x_start, y, cfg):
        model_kwargs = self._build_model_kwargs(y, cfg) 
        # 非拡散モデルの場合
        if self.diffuser_type is None:
            if self.use_ema:
                with torch.no_grad():
                    pred_x = self.ema(y, **model_kwargs)
            else:
                with torch.no_grad():
                    pred_x = self.model(y, **model_kwargs)
            return pred_x
        
        # 拡散モデル系統の場合
        if self.space == "latent":
            x_end = torch.randn(x_start.shape[0], self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]).to(self.device)
        else:
            x_end = torch.randn(x_start.shape[0],self.image_shape[0], self.image_shape[1], self.image_shape[2]).to(self.device) 
        if self.diffuser_type == "diffusion":
            pred_x_start = self.diffuser.ddim_sample_loop(
                self.ema if self.use_ema else self.model, # EMAを使うかどうか
                x_end.shape,
                noise=x_end,
                model_kwargs=model_kwargs,
                clip_denoised=True,
            )
        elif self.diffuser_type == "rectified_flow":
            pred_x_start = self.diffuser.sampler(
                self.ema if self.use_ema else self.model, 
                x_end.shape, 
                self.device, 
                model_kwargs=model_kwargs
            )
        if self.space == "latent":
            with torch.no_grad():
                pred_x_start = self.vae.decode(pred_x_start/self.scaling_factor).sample
        return pred_x_start
    
    # binaryとmericの計算
    def _calc_metric_binary(self, pred_mask, mask):
        num_channels = self.mask_shape[0]
        mask_np = mask.cpu().numpy()
        metrics = {}
        pred_binary_list = []
        for ch in range(num_channels):
            pred_mask_ch = pred_mask[:,ch]
            th = threshold_otsu(pred_mask_ch.cpu().numpy())
            pred_binary_mask_ch = (pred_mask_ch > th).float()
            pred_binary_list.append(pred_binary_mask_ch)
            metrics[f"ch{ch}"] = calc_metric(pred_binary_mask_ch.cpu().numpy(), mask_np[:,ch])
        pred_binary_mask = torch.stack(pred_binary_list, dim=1)
        return pred_binary_mask, metrics
        # if self.mask_shape[0] == 3:
        #     pred_mask_ch0 = pred_mask[:,0]
        #     pred_mask_ch1 = pred_mask[:,1]
        #     pred_mask_ch2 = pred_mask[:,2]
        #     th0 = threshold_otsu(pred_mask_ch0.cpu().numpy())
        #     th1 = threshold_otsu(pred_mask_ch1.cpu().numpy())
        #     th2 = threshold_otsu(pred_mask_ch2.cpu().numpy())
        #     pred_binary_mask_ch0 = (pred_mask_ch0 > th0).float()
        #     pred_binary_mask_ch1 = (pred_mask_ch1 > th1).float()
        #     pred_binary_mask_ch2 = (pred_mask_ch2 > th2).float()
        #     pred_binary_mask     = torch.stack([pred_binary_mask_ch0, pred_binary_mask_ch1, pred_binary_mask_ch2], dim=1)
        #     pred_binary_mask_ch0 = pred_binary_mask_ch0.cpu().numpy()
        #     pred_binary_mask_ch1 = pred_binary_mask_ch1.cpu().numpy()
        #     pred_binary_mask_ch2 = pred_binary_mask_ch2.cpu().numpy()
        #     mask = mask.cpu().numpy()
            
        #     metric_ch0 = calc_metric(pred_binary_mask_ch0, mask[:,0])
        #     metric_ch1 = calc_metric(pred_binary_mask_ch1, mask[:,1])
        #     metric_ch2 = calc_metric(pred_binary_mask_ch2, mask[:,2])
            
        # elif self.mask_shape[0] == 1:
        #     # 未実装:eroorで止めるようにする
        #     assert False, "mask_channels=1の二値セグメンテーションは未実装です"

            
  
    def train(self):
        
        for epoch in range(self.epochs):
            train_loss_list  = []
            test_SSIM_list   = [] # 拡散モデルの時はサンプリング過程がたくさん
            val_ch0_iou_list = []
            val_ch1_iou_list = []
            val_ch2_iou_list = []

            self.model.train()

            for batch in tqdm(self.train_loader): # バーがでてくる
                if self.task_select=="part1":
                    image = batch
                    patch_image, cond = create_crop(image.clone(), crop_size=self.crop_size)
                    x = patch_image.to(self.device)
                

                elif self.task_select=="part2":
                    image, mask = batch
                    patch_image, cond, patch_mask = create_crop(image.clone(),crop_size=self.crop_size,mask=mask.clone())
                    x = patch_mask.to(self.device)
                    patch_image = patch_image.to(self.device)

                cond = cond.to(self.device)
                t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
                
                if self.task_select=="part1":
                    model_kwargs = dict(cond=cond) # U-Netへの条件

                elif self.task_select=="part2":
                    with torch.no_grad():
                        cond_list = self.pos_encoder(cond)
                    model_kwargs = dict(cond=cond_list,patch_image=patch_image) # U-Netへの条件
                
                loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
                loss = loss_dict["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())
            
            
            self.model.eval()
            
            train_average_loss = sum(train_loss_list)/len(train_loss_list)
            # for batch in tqdm(self.val_loader):
            #     if self.task_select=="part1":
            #         image = batch
            #         patch_image, cond = create_crop(image.clone(), crop_size=self.crop_size)
            #         x = patch_image.to(self.device)

            #     elif self.task_select=="part2":
            #         image, mask = batch
            #         patch_image, cond, patch_mask = create_crop(image.clone(),crop_size=self.crop_size,mask=mask.clone())
            #         x = patch_mask.to(self.device)
            #         patch_image = patch_image.to(self.device)

            #     cond = cond.to(self.device)                
            #     if self.task_select=="part1":
            #         model_kwargs = dict(cond=cond) # U-Netへの条件

            #     elif self.task_select=="part2":
            #         with torch.no_grad():
            #             cond_list = self.pos_encoder(cond)
            #         model_kwargs = dict(cond=cond_list,patch_image=patch_image) # U-Netへの条件
                    
            #     pred_x = self.diffusion.ddim_sample_loop(
            #                     self.model,
            #                     x.shape,
            #                     model_kwargs=model_kwargs,
            #                     clip_denoised=True,
            #                 )
            #     break
            if (epoch+1)%25 == 0 or (epoch==0):
                with torch.no_grad():
                    
                    if self.task_select=="part1":
                        for image in tqdm(self.test_loader): # バーがでてくる
                            image = batch
                            patch_image, cond = create_crop(image.clone(), crop_size=self.crop_size)
                            x = patch_image.to(self.device)
                            cond = cond.to(self.device)                
                            model_kwargs = dict(cond=cond) # U-Netへの条件
                            pred_x = self.diffusion.ddim_sample_loop(
                                self.model, 
                                x.shape, 
                                model_kwargs = model_kwargs,
                                clip_denoised = True
                            )
                            # SSIMスコアの計算（SSIM損失は1からSSIMインデックスを引いたもの）
                            ssim = tgm.losses.SSIM(5, reduction='mean')
                            SSIM_loss = ssim(x, pred_x)               
                            test_SSIM_list.append(SSIM_loss.item())

                        test_average_SSIM = sum(test_SSIM_list) / len(test_SSIM_list)
                        if test_average_SSIM < self.best_score:
                            self.best_score = test_average_SSIM
                            save_model(self.model, 'best', self.dir_path)

                    elif self.task_select=="part2":
                        for batch in tqdm(self.val_loader):
                            image, mask = batch
                            
                            pred_ch0_list = []
                            pred_ch1_list = []
                            pred_ch2_list = []
                            # アンサンブル数: 5
                            for _ in range(5):
                                pred_x = torch.zeros(mask.shape) # cropを結合するようのマスクを用意
                                left_points = inference_image_split(mask,crop_size=self.crop_size)
                                for i in range(len(left_points)):
                                    patch_image, cond, patch_mask = create_inference_crop(image.clone(),crop_size=self.crop_size,mask=mask.clone(),left_point=left_points[i])
                                    x = patch_mask.to(self.device)
                                    patch_image = patch_image.to(self.device)
                                    cond = cond.to(self.device)                
                                    with torch.no_grad():
                                        cond_list = self.pos_encoder(cond)
                                    model_kwargs = dict(cond=cond_list,patch_image=patch_image) # U-Netへの条件
                                    pred_mini_x = self.diffusion.ddim_sample_loop(
                                            self.model, 
                                            x.shape, 
                                            model_kwargs = model_kwargs,
                                            clip_denoised = True
                                        )
                                    pred_x[:, :, left_points[i][0]:left_points[i][0]+self.crop_size, left_points[i][1]:left_points[i][1]+self.crop_size] = pred_mini_x

                                pred_mask = torch.sigmoid(pred_x)
                                pred_ch0_list.append(pred_mask[:,0]) # 背景
                                pred_ch1_list.append(pred_mask[:,1]) # 核
                                pred_ch2_list.append(pred_mask[:,2]) # 胚

                            # アンサンブルの平均を計算
                            mean_pred_ch0 = torch.mean(torch.stack(pred_ch0_list), dim=0) # 背景
                            mean_pred_ch1 = torch.mean(torch.stack(pred_ch1_list), dim=0) # 核
                            mean_pred_ch2 = torch.mean(torch.stack(pred_ch2_list), dim=0) # 胚

                            # 二値化の閾値を計算 (例: Otsu)
                            threshold_ch0 = threshold_otsu(mean_pred_ch0.cpu().numpy())
                            threshold_ch1 = threshold_otsu(mean_pred_ch1.cpu().numpy())
                            threshold_ch2 = threshold_otsu(mean_pred_ch2.cpu().numpy())

                            # 各チャネルを二値化
                            pred_binary_ch0 = (mean_pred_ch0 > threshold_ch0).float()
                            pred_binary_ch1 = (mean_pred_ch1 > threshold_ch1).float()
                            pred_binary_ch2 = (mean_pred_ch2 > threshold_ch2).float()
                            # 3チャネルに結合
                            pred_binary_mask = torch.stack([pred_binary_ch0, pred_binary_ch1, pred_binary_ch2], dim=1)  # [N, 3, H, W]
                            mask            = mask.cpu().numpy()
                                
                            for i in range(mask.shape[0]):
                                val_ch0_iou_list.append(jaccard(pred_binary_ch0[i].cpu().numpy(), mask[i][0]))
                                val_ch1_iou_list.append(jaccard(pred_binary_ch1[i].cpu().numpy(), mask[i][1]))
                                val_ch2_iou_list.append(jaccard(pred_binary_ch2[i].cpu().numpy(), mask[i][2]))



                        val_ch0_iou_averege = sum(val_ch0_iou_list)/len(val_ch0_iou_list) # 背景
                        val_ch1_iou_averege = sum(val_ch1_iou_list)/len(val_ch1_iou_list) # 核
                        val_ch2_iou_averege = sum(val_ch2_iou_list)/len(val_ch2_iou_list) # 胚
                        val_all_iou_average = (val_ch1_iou_averege+val_ch2_iou_averege)/2 # 核と胚の平均


                        if val_all_iou_average > self.best_score:
                            self.best_score = val_all_iou_average
                            save_model(self.model, 'best', self.dir_path)                    

                if self.wandb_flag:
                    if self.task_select=="part1": 
                        send_num_image = min(5, image.shape[0]) # 画像を送るときの最大数
                        wandb_image=[wandb.Image(image[i]) for i in range(send_num_image)]
                        wandb_x    =[wandb.Image(x[i]) for i in range(send_num_image)]
                        wandb_pred_x    =[wandb.Image(pred_x[i]) for i in range(send_num_image)]  
                        wandb_cond=[wandb.Image(cond[i]) for i in range(send_num_image)]
                        wandb.log({
                            'train_loss':train_average_loss,
                            'test_score':test_average_SSIM,
                            'image':wandb_image,
                            'x':wandb_x,
                            'pred_x':wandb_pred_x,
                            'cond':wandb_cond,
                        })
                            
                    elif self.task_select=="part2":
                        wandb_image=[wandb.Image(image[i]) for i in range(len(image))]
                        wandb_mask =[wandb.Image(mask[i].transpose(1, 2, 0)) for i in range(len(mask))]
                        wandb_x    =[wandb.Image(x[i]) for i in range(len(x))]
                        wandb_pred_mask    =[wandb.Image(pred_binary_mask[i]) for i in range(len(pred_x))]  
                        wandb_cond=[wandb.Image(cond[i]) for i in range(len(cond))]
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
            else:
                if self.wandb_flag:
                    if self.task_select=="part1":
                        wandb_image=[wandb.Image(image[i]) for i in range(len(image))]
                        wandb_x    =[wandb.Image(x[i]) for i in range(len(x))]
                        wandb_pred_x    =[wandb.Image(pred_x[i]) for i in range(len(pred_x))]  
                        wandb_cond=[wandb.Image(cond[i]) for i in range(len(cond))]
                        wandb.log({
                            'train_loss':train_average_loss,
                            # 'image':wandb_image,
                            # 'x':wandb_x,
                            # 'pred_x':wandb_pred_x,
                            # 'cond':wandb_cond,
                        })
                    elif self.task_select=="part2":
                        # wandb_image=[wandb.Image(image[i]) for i in range(len(image))]
                        # wandb_mask =[wandb.Image(mask[i]) for i in range(len(mask))]
                        # wandb_patch_image = [wandb.Image(patch_image[i]) for i in range(len(patch_image))]
                        # wandb_x    =[wandb.Image(x[i]) for i in range(len(x))]
                        # wandb_pred_x    =[wandb.Image(pred_x[i]) for i in range(len(pred_x))]  
                        # wandb_cond=[wandb.Image(cond[i]) for i in range(len(cond))]
                        wandb.log({
                            'train_loss':train_average_loss,
                            # 'image':wandb_image,
                            # 'mask':wandb_mask,
                            # 'patch_image': wandb_patch_image,
                            # 'x':wandb_x,
                            # 'pred_x':wandb_pred_x,
                            # 'cond':wandb_cond,
                        })  
        save_model(self.model,'last', self.dir_path)
        
    def test(self):
        # Testの開始
        print("start test")
        self.model.eval()
        test_ch0_iou_list = []
        test_ch1_iou_list = []
        test_ch2_iou_list = []
        if self.task_select=="part2":
            for batch in tqdm(self.test_loader):
                image, mask = batch
                            
                pred_ch0_list = []
                pred_ch1_list = []
                pred_ch2_list = []
                # アンサンブル数: 15
                for _ in range(5):
                    pred_x = torch.zeros(mask.shape) # cropを結合するようのマスクを用意
                    left_points = inference_image_split(mask,crop_size=self.crop_size)
                    for i in range(len(left_points)):
                        patch_image, cond, patch_mask = create_inference_crop(image.clone(),crop_size=self.crop_size,mask=mask.clone(),left_point=left_points[i])
                        x = patch_mask.to(self.device)
                        patch_image = patch_image.to(self.device)
                        cond = cond.to(self.device)                
                        with torch.no_grad():
                            cond_list = self.pos_encoder(cond)
                        model_kwargs = dict(cond=cond_list,patch_image=patch_image) # U-Netへの条件
                        pred_mini_x = self.diffusion.ddim_sample_loop(
                            self.model, 
                            x.shape, 
                            model_kwargs = model_kwargs,
                            clip_denoised = True
                        )
                        pred_x[:, :, left_points[i][0]:left_points[i][0]+self.crop_size, left_points[i][1]:left_points[i][1]+self.crop_size] = pred_mini_x

                    pred_mask = torch.sigmoid(pred_x)
                    pred_ch0_list.append(pred_mask[:,0]) # 背景
                    pred_ch1_list.append(pred_mask[:,1]) # 核
                    pred_ch2_list.append(pred_mask[:,2]) # 胚

                # アンサンブルの平均を計算
                mean_pred_ch0 = torch.mean(torch.stack(pred_ch0_list), dim=0) # 背景
                mean_pred_ch1 = torch.mean(torch.stack(pred_ch1_list), dim=0) # 核
                mean_pred_ch2 = torch.mean(torch.stack(pred_ch2_list), dim=0) # 胚

                # 二値化の閾値を計算 (例: Otsu)
                threshold_ch0 = threshold_otsu(mean_pred_ch0.cpu().numpy())
                threshold_ch1 = threshold_otsu(mean_pred_ch1.cpu().numpy())
                threshold_ch2 = threshold_otsu(mean_pred_ch2.cpu().numpy())

                # 各チャネルを二値化
                pred_binary_ch0 = (mean_pred_ch0 > threshold_ch0).float()
                pred_binary_ch1 = (mean_pred_ch1 > threshold_ch1).float()
                pred_binary_ch2 = (mean_pred_ch2 > threshold_ch2).float()
                # 3チャネルに結合
                pred_binary_mask = torch.stack([pred_binary_ch0, pred_binary_ch1, pred_binary_ch2], dim=1)  # [N, 3, H, W]
                mask    = mask.cpu().numpy()
                for i in range(mask.shape[0]):
                    test_ch0_iou_list.append(jaccard(pred_binary_ch0[i].cpu().numpy(), mask[i,0]))
                    test_ch1_iou_list.append(jaccard(pred_binary_ch1[i].cpu().numpy(), mask[i,1]))
                    test_ch2_iou_list.append(jaccard(pred_binary_ch2[i].cpu().numpy(), mask[i,2]))

            test_ch0_iou_averege = sum(test_ch0_iou_list)/len(test_ch0_iou_list) # 背景
            test_ch1_iou_averege = sum(test_ch1_iou_list)/len(test_ch1_iou_list) # 核
            test_ch2_iou_averege = sum(test_ch2_iou_list)/len(test_ch2_iou_list) # 胚
            test_all_iou_average = (test_ch1_iou_averege+test_ch2_iou_averege)/2 # 核と胚の平均
                            
            if self.task_select=="part2":
                wandb_image=[wandb.Image(image[i]) for i in range(len(image))]
                wandb_mask =[wandb.Image(mask[i].transpose(1, 2, 0)) for i in range(len(mask))]
                wandb_pred_mask    =[wandb.Image(pred_binary_mask[i]) for i in range(len(x))]
                wandb.log({
                    'test_背景IoU': test_ch0_iou_averege,
                    'test_核IoU': test_ch1_iou_averege,
                    'test_胚IoU': test_ch2_iou_averege,
                    'test_胚と核の平均IoU': test_all_iou_average,

                    'image':wandb_image,
                    'mask':wandb_mask,
                    "pred mask": wandb_pred_mask,
                })

        print(f"ch0 iou: {test_ch0_iou_averege}")
        print(f"ch1 iou: {test_ch1_iou_averege}")
        print(f"ch2 iou: {test_ch2_iou_averege}")
        print(f"all iou: {test_all_iou_average}")

        wandb.finish()