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
        self.test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=False)

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
        
        self.best_path            = os.path.join(dir_path,"weights",f"weight_epoch_best.pth")
        self.ema_best_path        = os.path.join(dir_path,"weights",f"weight_epoch_best_ema.pth")
        self.last_path            = os.path.join(dir_path,"weights",f"weight_epoch_last.pth")
        self.ema_last_path        = os.path.join(dir_path,"weights",f"weight_epoch_last_ema.pth")

        self.image_shape  = (cfg["data"]["img_channels"], cfg["data"]["img_size"], cfg["data"]["img_size"])
        self.mask_shape   = (cfg["data"]["mask_channels"], cfg["data"]["img_size"], cfg["data"]["img_size"])
        if self.space == "latent":
            self.latent_shape = (cfg["vae"]["latent"]["latent_dim"], cfg["vae"]["latent"]["latent_size"], cfg["vae"]["latent"]["latent_size"])
    
    def vae_check(self,cfg):
        tags = [cfg["model"]["model_name"], cfg["data"]["dataset"], cfg["diffuser_type"], cfg["space"]]
        tags = [t for t in tags if t is not None]
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["run_name"],
            tags=tags,
            config=cfg,
            save_code=cfg["wandb"]["save_code"],
        )
        for batch in self.train_loader:
            image, mask = batch
            x_start = mask.to(self.device)
            with torch.no_grad():
                x_start = self.vae.encode(x=x_start).latent_dist
                if cfg["vae"]["latent"]["latent_sample"] == "mean":
                    x_start = x_start.mode()
                elif cfg["vae"]["latent"]["latent_sample"] == "sample":
                    x_start = x_start.sample()
                x_start = x_start.mul_(self.scaling_factor)
                x_start = self.vae.decode(x_start/self.scaling_factor).sample
            break
        pred_binary_mask, metrics = self._calc_metric_binary(x_start, mask)
        x_start = pred_binary_mask.cpu().numpy()
        mask = mask.cpu().numpy()
        
        wandb_mask = [wandb.Image(mask[i].transpose(1, 2, 0)) for i in range(mask.shape[0])]
        wandb_x_start = [wandb.Image(x_start[i].transpose(1, 2, 0)) for i in range(x_start.shape[0])]
        wandb.log({
            "mask": wandb_mask,
            "x_start": wandb_x_start,
            "metrics": metrics,
        })
        wandb.finish()

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
        
        if cfg["wandb"]["pred_feature"]:
            pred_mask         = pred_mask.cpu().numpy()
            wandb_pred_ch1_x  = [wandb.Image(pred_mask[i][1]) for i in range(wandb_num_images)]
            wandb_pred_ch2_x  = [wandb.Image(pred_mask[i][2]) for i in range(wandb_num_images)]
            log_dict.update({
                "pred_ch1_x": wandb_pred_ch1_x,
                "pred_ch2_x": wandb_pred_ch2_x,
            })

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
      
    def test_loop(self, cfg):
        # test_loopは一枚ずつ画像のスコアを計測
        # 最後に平均も送信
        # best modelを読み込む
        if cfg["test"]["use_best_model"]:
            self.model = load_model(self.model, self.best_path)
            if self.use_ema:
                print("Reading the best EMA Model...")
                self.ema = load_model(self.ema, self.ema_best_path)
        else:
            self.model = load_model(self.model, self.last_path)
            if self.use_ema:
                print("Reading the last EMA Model...")
                print(self.ema_last_path)
                self.ema = load_model(self.ema, self.ema_last_path)
        tags = ["test", cfg["model"]["model_name"], cfg["data"]["dataset"], cfg["diffuser_type"], cfg["space"]]
        tags = [t for t in tags if t is not None]
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["run_name"],
            tags=tags,
            config=cfg,
            save_code=cfg["wandb"]["save_code"],
        )
        all_metrics = defaultdict(list)
        for batch in tqdm(self.test_loader, desc="Testing", disable=self.rank != 0):
            image, mask = batch
            x_start = mask.to(self.device)
            y       = image.to(self.device)
            pred_x_start_list = []
            if self.diffuser_type is None:
                pred_mask = self._forward_inference(x_start, y, cfg)
                std_mask  = None
            else:
                for _ in range(cfg["diffusion"]["test_ensemble"]):
                    pred_x_start = self._forward_inference(x_start, y, cfg)
                    pred_x_start_list.append(pred_x_start)
                # 平均, 標準偏差を計算
                pred_mask   = torch.mean(torch.stack(pred_x_start_list), dim=0) # emsembleの平均
                std_mask    = torch.std(torch.stack(pred_x_start_list), dim=0)  # emsembleの標準偏差
            pred_binary_mask, metrics = self._calc_metric_binary(pred_mask, mask)
            for ch, metric in metrics.items():
                for key, value in metric.items():
                    all_metrics[f"{ch}_{key}"].append(value)
            # miouも計算
            metrics["miou"] = (metrics["ch1"]["iou"] + metrics["ch2"]["iou"]) / 2
            all_metrics["miou"].append(metrics["miou"])
            # wandbへここで送信 (batchは1で固定されてるから)
            y = y.cpu().numpy()
            mask = mask.cpu().numpy()
            pred_binary_mask = pred_binary_mask.cpu().numpy()
            wandb_y           = [wandb.Image(y[i].transpose(1, 2, 0)) for i in range(y.shape[0])]
            wandb_x           = [wandb.Image(mask[i].transpose(1, 2, 0)) for i in range(mask.shape[0])]
            pred_binary_x     = [wandb.Image(pred_binary_mask[i].transpose(1, 2, 0)) for i in range(pred_binary_mask.shape[0])]
            log_dict = {
                "y": wandb_y,
                "mask": wandb_x,
                "pred_binary_mask": pred_binary_x,
            }
            log_dict.update(metrics)
            wandb.log(log_dict)
        avg_metrics = {k: sum(v) / self.test_size for k, v in all_metrics.items()}
        # test_プレフィックスをつける
        test_avg_metrics = {f"test_{k}": v for k, v in avg_metrics.items()}
        wandb.log(
            test_avg_metrics
        )
        wandb.finish()


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