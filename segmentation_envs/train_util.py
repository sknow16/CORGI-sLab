from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import wandb
from skimage.filters import threshold_otsu
from metric.metric import jaccard
from guided_diffusion.save_log import save_model
import matplotlib.pyplot as plt
import numpy as np
import random
# from pytorch_msssim import ssim
import torchgeometry as tgm
from guided_diffusion.save_log import load_model
x = torch.randn(1,1,32,32)
mask = torch.zeros(8,8)

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
            pos_encoder,
            diffusion,
            optimizer,
            criterion,
            train_set,
            val_set,
            test_set=None,
            args,
            dir_path,
            scheduler=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.check_point:
            model = load_model(model, args.check_point_path)
            print("check pointから追加学習中")
        self.model  = model.to(self.device)
        if pos_encoder is not None:
            self.pos_encoder = pos_encoder.to(self.device)
            self.pos_encoder.eval()
        else:
            self.pos_encoder = None
        self.task_select = args.task_select

        self.diffusion = diffusion
        
        self.optimizer = optimizer
        self.criterion = criterion # 今回は使ってない(普通の3DU-Netのときはつかう)

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=True)
        self.val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        
        self.crop_size   = args.crop_size # 64
        self.epochs = args.epochs
        self.test_size = args.test_size
        self.val_size  = args.val_size
        self.wandb_flag = args.wandb_flag
        self.dir_path = dir_path
        if self.task_select=="part1":
            self.best_score = float('inf')
        elif self.task_select=="part2":
            self.best_score = 0
            
        if self.wandb_flag:
            wandb.init(
                name=args.model_name,
                project=args.project_name,
                tags=[args.dim_size, args.task_select],
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
                        wandb_image=[wandb.Image(image[i]) for i in range(len(image))]
                        wandb_x    =[wandb.Image(x[i]) for i in range(len(x))]
                        wandb_pred_x    =[wandb.Image(pred_x[i]) for i in range(len(pred_x))]  
                        wandb_cond=[wandb.Image(cond[i]) for i in range(len(cond))]
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