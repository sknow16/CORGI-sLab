import os
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import wandb
from skimage.filters import threshold_otsu
from metric.metric import jaccard, dice, hausdorff_distance_95
from save_log import save_model, load_model

import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # GPUの乱数も固定（必要なら）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # マルチGPU用
        torch.backends.cudnn.deterministic = True  # CuDNNの動作を固定
        torch.backends.cudnn.benchmark = False  # 再現性を優先

class Trainer:
    def __init__(
            self,
            *, # *以降は呼び出す際にキーワード引数で指定する必要がある
            model,
            diffusion,
            optimizer,
            criterion,
            train_set,
            val_set,
            test_set=None,
            args,
            dir_path,
            num_params,
            num_trainable_params,
            scheduler=None,
            
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.diffusion = diffusion
        self.dir_path = dir_path
        self.seed      = args.seed
        self.optimizer = optimizer
        self.criterion = criterion # 今回は使ってない(普通の3DU-Netのときはつかう)

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=True)
        self.val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        
        self.epochs = args.epochs
        self.test_size = args.test_size
        self.val_size  = args.val_size
        self.wandb_flag = args.wandb_flag
        self.best_mIoU = 0.0
        # if self.wandb_flag:
        #     wandb.init(
        #         project=args.project_name,
        #         name=args.model_name,
        #         tags=[args.dim_size],
        #         notes = args.model_detail,
        #         config={
        #         "model":         args.model_name,
        #         "dataset":       args.dataset,
        #         "epochs":        args.epochs,
        #         "image_size":    args.img_size,
        #         "in_channel":    args.in_channels,
        #         "out_channel":   args.out_channels,

        #         "batch_size":    args.batch_size,
        #         "learning_rate": args.lr,

        #         "optimizer": args.optimizer,         # 使用している最適化手法
        #         "loss_function": args.loss_function, # 使用している損失関数

        #         "train_volume": args.train_size,
        #         "val_volume": args.val_size,
        #         "test_volume": args.test_size,
        #         "num_params": num_params,
        #         "num_trainable_params": num_trainable_params,
        #         }
        #     )
    
    def train(self):
        set_seed(self.seed)
        for epoch in range(self.epochs):
            train_loss_list = []
            val_loss_list = [] # 拡散モデルの時はサンプリング過程がたくさん
            self.model.train()

            for batch in tqdm(self.train_loader): # バーがでてくる
                image = batch['image'].to(self.device)
                mask = batch['label'].to(self.device)
                t = torch.randint(0, self.diffusion.num_timesteps, (mask.shape[0],), device=self.device)
                model_kwargs = dict(image=image)
                loss_dict = self.diffusion.training_losses(self.model, mask, t, model_kwargs)
                loss = loss_dict["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())

            self.model.eval()
            jaccard_0_list = []
            jaccard_1_list = []
            jaccard_2_list = []
            if (epoch+1) % 10!=0: 
                train_average_loss = sum(train_loss_list)/len(train_loss_list)
                wandb.log({
                        'train_loss':train_average_loss,
                })
            elif (epoch+1) % 10==0:
                for batch in tqdm(self.val_loader):
                    image = batch['image'].to(self.device)
                    mask = batch['label'].to(self.device)
                    model_kwargs = dict(image=image)
                    pred_mask_list = []

                    # アンサンブル数 5
                    for _ in range(10):
                        with torch.no_grad():
                            pred_mask = self.diffusion.ddim_sample_loop(
                                self.model,
                                mask.shape,
                                model_kwargs=model_kwargs,
                                clip_denoised=True,
                            )
                        pred_mask = torch.sigmoid(pred_mask)
                        pred_mask_list.append(pred_mask)
                    mean_pred_mask = torch.mean(torch.stack(pred_mask_list),dim=0)
                    pred_mask_0 = mean_pred_mask[:,0,:,:,:]
                    pred_mask_1 = mean_pred_mask[:,1,:,:,:]
                    pred_mask_2 = mean_pred_mask[:,2,:,:,:]
                    th0 = threshold_otsu(pred_mask_0.cpu().numpy())
                    th1 = threshold_otsu(pred_mask_1.cpu().numpy())
                    th2 = threshold_otsu(pred_mask_2.cpu().numpy())
                    pred_mask_0_binary = (pred_mask_0>th0).float().cpu().numpy()
                    pred_mask_1_binary = (pred_mask_1>th1).float().cpu().numpy()
                    pred_mask_2_binary = (pred_mask_2>th2).float().cpu().numpy()
                                
                    mask_0 = mask[:,0,:,:,:].cpu().numpy()
                    mask_1 = mask[:,1,:,:,:].cpu().numpy()
                    mask_2 = mask[:,2,:,:,:].cpu().numpy()

                    jaccard_0_score = jaccard(pred_mask_0_binary, mask_0)
                    jaccard_1_score = jaccard(pred_mask_1_binary, mask_1)
                    jaccard_2_score = jaccard(pred_mask_2_binary, mask_2)

                    jaccard_0_list.append(jaccard_0_score*len(mask))
                    jaccard_1_list.append(jaccard_1_score*len(mask))
                    jaccard_2_list.append(jaccard_2_score*len(mask))

                jaccard_0_average = sum(jaccard_0_list)/self.val_size
                jaccard_1_average = sum(jaccard_1_list)/self.val_size
                jaccard_2_average = sum(jaccard_2_list)/self.val_size
                jaccard_average = (jaccard_0_average + jaccard_1_average + jaccard_2_average)/3.0
                train_average_loss = sum(train_loss_list)/len(train_loss_list)
                if jaccard_average > self.best_mIoU:
                    self.best_mIoU = jaccard_average
                    save_model(self.model, "best", self.dir_path)
                    print(f"モデルを保存しました。mIoU: {self.best_mIoU:.4f} Epoch: {epoch+1}")

                if self.wandb_flag:
                    mean_pred_mask = mean_pred_mask.cpu().numpy()
                    image = image.cpu().numpy()
                    mask  = mask.cpu().numpy()
                    wandb_pred_mask0 = [wandb.Image(mean_pred_mask[i,0,:,:,32]) for i in range(len(pred_mask))]
                    wandb_pred_mask1 = [wandb.Image(mean_pred_mask[i,1,:,:,32]) for i in range(len(pred_mask))]
                    wandb_pred_mask2 = [wandb.Image(mean_pred_mask[i,2,:,:,32]) for i in range(len(pred_mask))]
                    wandb_mask0 = [wandb.Image(mask[i,0,:,:,32]) for i in range(len(mask))]
                    wandb_mask1 = [wandb.Image(mask[i,1,:,:,32]) for i in range(len(mask))]
                    wandb_mask2 = [wandb.Image(mask[i,2,:,:,32]) for i in range(len(mask))]
                    wandb_image0 = [wandb.Image(image[i,0,:,:,32]) for i in range(len(image))]
                    wandb_image1 = [wandb.Image(image[i,1,:,:,32]) for i in range(len(image))]
                    wandb_image2 = [wandb.Image(image[i,2,:,:,32]) for i in range(len(image))]
                    wandb_image3 = [wandb.Image(image[i,3,:,:,32]) for i in range(len(image))]
                    wandb_pred_mask_binary0 = [wandb.Image(pred_mask_0_binary[i,:,:,32]) for i in range(len(pred_mask_0_binary))]
                    wandb_pred_mask_binary1 = [wandb.Image(pred_mask_1_binary[i,:,:,32]) for i in range(len(pred_mask_1_binary))]
                    wandb_pred_mask_binary2 = [wandb.Image(pred_mask_2_binary[i,:,:,32]) for i in range(len(pred_mask_2_binary))]

                    wandb.log({
                        'train_loss':train_average_loss,
                        'pred_mask0':wandb_pred_mask0,
                        'pred_mask1':wandb_pred_mask1,
                        'pred_mask2':wandb_pred_mask2,
                        'pred_mask_binary0':wandb_pred_mask_binary0,
                        'pred_mask_binary1':wandb_pred_mask_binary1,
                        'pred_mask_binary2':wandb_pred_mask_binary2,
                        'mask0':wandb_mask0,
                        'mask1':wandb_mask1,
                        'mask2':wandb_mask2,
                        'IoU_ch0':jaccard_0_average,
                        'IoU_ch1':jaccard_1_average,
                        'IoU_ch2':jaccard_2_average,
                        'mIoU':jaccard_average,
                        'image0':wandb_image0,
                        'image1':wandb_image1,
                        'image2':wandb_image2,
                        'image3':wandb_image3,
                    })
        
        save_model(self.model, "last", self.dir_path)
        wandb.finish()
    def test(self, args):
        set_seed(self.seed)
        wandb.init(
                project=args.project_name+"_test",
                name=args.model_name,
                tags=[args.dim_size],
                notes= args.model_detail,
                config={
                "model":         args.model_name,
                "dataset":       args.dataset,
                "epochs":        args.epochs,
                "image_size":    args.img_size,
                "in_channel":    args.in_channels,
                "out_channel":   args.out_channels,

                "batch_size":    args.batch_size,
                "learning_rate": args.lr,

                "optimizer": args.optimizer,         # 使用している最適化手法
                "loss_function": args.loss_function, # 使用している損失関数

                "train_volume": args.train_size,
                "val_volume": args.val_size,
                "test_volume": args.test_size
                })
        parameters = sum(p.numel() for p in self.model.parameters())
        print(f"モデルのパラメータ数: {parameters/1e6:.2f} Million")
        # self.model = load_model(self.model, os.path.join(self.dir_path,"weights","weight_epoch_best.pth"))
        self.model = load_model(self.model, os.path.join("/root/save/log/20251111:Diff-UNet.ref_epochs_500:Brats2021/weights/weight_epoch_best.pth"))
        # parameters = sum(p.numel() for p in self.model.parameters())
        parameters = sum(p.numel() for p in self.model.parameters())
        print(f"モデルのパラメータ数: {parameters/1e6:.2f} Million")
        # 学習可能なパラメータ数を計算
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"学習可能なパラメータ数: {num_trainable_params/1e6:.2f} Million")
        
        self.model.eval()
        jaccard_0_list = []
        jaccard_1_list = []
        jaccard_2_list = []
        dice_0_list = []
        dice_1_list = []
        dice_2_list = []
        hausdorff_0_list = []
        hausdorff_1_list = []
        hausdorff_2_list = []
        for batch in tqdm(self.test_loader):
            image = batch['image'].to(self.device)
            mask = batch['label'].to(self.device)
            model_kwargs = dict(image=image)
            pred_mask_list = []

            # アンサンブル数 
            for _ in range(10):
                with torch.no_grad():
                    pred_mask = self.diffusion.ddim_sample_loop(
                        self.model,
                        mask.shape,
                        model_kwargs=model_kwargs,
                        clip_denoised=True,
                    )
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask_list.append(pred_mask)
            mean_pred_mask = torch.mean(torch.stack(pred_mask_list),dim=0)
            # std_pred_mask  = torch.std(torch.stack(pred_mask_list),dim=0)
            pred_mask_0 = mean_pred_mask[:,0,:,:,:]
            pred_mask_1 = mean_pred_mask[:,1,:,:,:]
            pred_mask_2 = mean_pred_mask[:,2,:,:,:]
            th0 = threshold_otsu(pred_mask_0.cpu().numpy())
            th1 = threshold_otsu(pred_mask_1.cpu().numpy())
            th2 = threshold_otsu(pred_mask_2.cpu().numpy())
            pred_mask_0_binary = (pred_mask_0>th0).float().cpu().numpy()
            pred_mask_1_binary = (pred_mask_1>th1).float().cpu().numpy()
            pred_mask_2_binary = (pred_mask_2>th2).float().cpu().numpy()
                        
            mask_0 = mask[:,0,:,:,:].cpu().numpy()
            mask_1 = mask[:,1,:,:,:].cpu().numpy()
            mask_2 = mask[:,2,:,:,:].cpu().numpy()

            jaccard_0_score = jaccard(pred_mask_0_binary, mask_0)
            jaccard_1_score = jaccard(pred_mask_1_binary, mask_1)
            jaccard_2_score = jaccard(pred_mask_2_binary, mask_2)

            dice_0_score = dice(pred_mask_0_binary, mask_0)
            dice_1_score = dice(pred_mask_1_binary, mask_1)
            dice_2_score = dice(pred_mask_2_binary, mask_2)

            hausdorff_0_score = hausdorff_distance_95(pred_mask_0_binary, mask_0)
            hausdorff_1_score = hausdorff_distance_95(pred_mask_1_binary, mask_1)
            hausdorff_2_score = hausdorff_distance_95(pred_mask_2_binary, mask_2)

            mIoU = (jaccard_0_score + jaccard_1_score + jaccard_2_score)/3.0
            jaccard_0_list.append(jaccard_0_score*len(mask))
            jaccard_1_list.append(jaccard_1_score*len(mask))
            jaccard_2_list.append(jaccard_2_score*len(mask))
            dice_0_list.append(dice_0_score*len(mask))
            dice_1_list.append(dice_1_score*len(mask))
            dice_2_list.append(dice_2_score*len(mask))
            hausdorff_0_list.append(hausdorff_0_score*len(mask))
            hausdorff_1_list.append(hausdorff_1_score*len(mask))
            hausdorff_2_list.append(hausdorff_2_score*len(mask))

            image = image.cpu().numpy()
            mask  = mask.cpu().numpy()

            wandb_mask0 = [wandb.Image(mask[i,0,:,:,32]) for i in range(len(mask))]
            wandb_mask1 = [wandb.Image(mask[i,1,:,:,32]) for i in range(len(mask))]
            wandb_mask2 = [wandb.Image(mask[i,2,:,:,32]) for i in range(len(mask))]
            wandb_image0 = [wandb.Image(image[i,0,:,:,32]) for i in range(len(image))]
            wandb_image1 = [wandb.Image(image[i,1,:,:,32]) for i in range(len(image))]
            wandb_image2 = [wandb.Image(image[i,2,:,:,32]) for i in range(len(image))]
            wandb_image3 = [wandb.Image(image[i,3,:,:,32]) for i in range(len(image))]
            wandb_pred_mask_binary0 = [wandb.Image(pred_mask_0_binary[i,:,:,32]) for i in range(len(pred_mask_0_binary))]
            wandb_pred_mask_binary1 = [wandb.Image(pred_mask_1_binary[i,:,:,32]) for i in range(len(pred_mask_1_binary))]
            wandb_pred_mask_binary2 = [wandb.Image(pred_mask_2_binary[i,:,:,32]) for i in range(len(pred_mask_2_binary))]
            wandb.log({
                    'pred_mask_binary0':wandb_pred_mask_binary0,
                    'pred_mask_binary1':wandb_pred_mask_binary1,
                    'pred_mask_binary2':wandb_pred_mask_binary2,
                    'mask0':wandb_mask0,
                    'mask1':wandb_mask1,
                    'mask2':wandb_mask2,
                    'mIoU' : mIoU,
                    'IoU_ch0' :jaccard_0_score,
                    'IoU_ch1' :jaccard_1_score,
                    'IoU_ch2' :jaccard_2_score,
                    "Dice_ch0" : dice_0_score,
                    "Dice_ch1" : dice_1_score,
                    "Dice_ch2" : dice_2_score,
                    "Hausdorff_ch0" : hausdorff_0_score,
                    "Hausdorff_ch1" : hausdorff_1_score,
                    "Hausdorff_ch2" : hausdorff_2_score,
                    'image0':wandb_image0,
                    'image1':wandb_image1,
                    'image2':wandb_image2,
                    'image3':wandb_image3,
                })
            
        jaccard_0_average = sum(jaccard_0_list)/self.test_size
        jaccard_1_average = sum(jaccard_1_list)/self.test_size
        jaccard_2_average = sum(jaccard_2_list)/self.test_size
        jaccard_average = (jaccard_0_average + jaccard_1_average + jaccard_2_average)/3.0
        dice_0_average = sum(dice_0_list)/self.test_size
        dice_1_average = sum(dice_1_list)/self.test_size
        dice_2_average = sum(dice_2_list)/self.test_size
        hausdorff_0_average = sum(hausdorff_0_list)/self.test_size
        hausdorff_1_average = sum(hausdorff_1_list)/self.test_size
        hausdorff_2_average = sum(hausdorff_2_list)/self.test_size
        if self.wandb_flag:
            wandb.log({
                    'test_mIoU' :jaccard_average,
                    'test_IoU_ch0' :jaccard_0_average,
                    'test_IoU_ch1' :jaccard_1_average,
                    'test_IoU_ch2' :jaccard_2_average,
                    'test_dice_ch0' :dice_0_average,
                    'test_dice_ch1' :dice_1_average,
                    'test_dice_ch2' :dice_2_average,
                    'test_hausdorff_ch0' :hausdorff_0_average,
                    'test_hausdorff_ch1' :hausdorff_1_average,
                    'test_hausdorff_ch2' :hausdorff_2_average,
                })
        wandb.finish()