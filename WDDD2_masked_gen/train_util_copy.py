from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import wandb
from skimage.filters import threshold_otsu
from metric.metric import jaccard
from guided_diffusion.save_log import save_model, load_model
import matplotlib.pyplot as plt
import numpy as np
import random
# from pytorch_msssim import ssim
import torchgeometry as tgm
x = torch.randn(1,1,32,32)
mask = torch.zeros(8,8)

def random_coordinate(mask_point_range, batch):
    coordinates = [(random.randint(0, mask_point_range), random.randint(0, mask_point_range)) for _ in range(batch)]
    return coordinates

batch = 1
mask_point_range = 24

# random_coordinates = random_coordinate(mask_point_range, batch)
# print(random_coordinates)

def create_crop(x, crop_size):
    n, c, h, w = x.shape
    left_points = random_coordinate(w-crop_size, n)
    mini_x = torch.zeros(n, c, crop_size, crop_size)
    for i in range(n):
        mini_x[i] = x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]
        x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size] = 0
    return mini_x, x

# mini_x, x = create_crop(x, 4)
    
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
            scheduler=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.diffusion = diffusion
        
        self.optimizer = optimizer
        self.criterion = criterion # 今回は使ってない(普通の3DU-Netのときはつかう)

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=True)
        self.val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        
        self.epochs = args.epochs
        self.test_size = args.test_size
        self.val_size  = args.val_size
        self.wandb_flag = args.wandb_flag
        self.dir_path = dir_path
        self.best_score = float('inf')
        if self.wandb_flag:
            wandb.init(
                project=args.project_name,
                tags=[args.dim_size],
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
                }
            )
    
    def train(self):
        
        for epoch in range(self.epochs):
            train_loss_list = []
            test_SSIM_list = [] # 拡散モデルの時はサンプリング過程がたくさん
            model_path = "/root/save/log/masked_gen_epochs_200:Brats/weights/weight_epoch_best.pth"
            self.model = load_model(self.model, model_path)

            # self.model.train()

            # for image in tqdm(self.train_loader): # バーがでてくる
            #     x, cond = create_crop(image, crop_size=64)
            #     x = x.to(self.device)
            #     cond = cond.to(self.device)
            #     t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
            #     model_kwargs = dict(cond=cond)
            #     loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
            #     loss = loss_dict["loss"]
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            #     train_loss_list.append(loss.item())
            
            
            self.model.eval()
            
            # train_average_loss = sum(train_loss_list)/len(train_loss_list)
            train_average_loss = 0
            for image in tqdm(self.val_loader):
                x, cond = create_crop(image, crop_size=64)
                x = x.to(self.device)
                cond = cond.to(self.device)
                model_kwargs = dict(cond=cond)
                pred_x, cond_feature = self.diffusion.ddim_sample_loop(
                                self.model,
                                x.shape,
                                model_kwargs=model_kwargs,
                                clip_denoised=True,
                            )
                break
            print(pred_x.shape)
            print(cond_feature.shape)

            # if (epoch+1)%3 == 0:
            #     with torch.no_grad():
            #         for image in tqdm(self.test_loader): # バーがでてくる
            #             x, cond = create_crop(image, crop_size=64)
            #             x = x.to(self.device)
            #             x_T = torch.randn_like(x).to(self.device)
            #             cond = cond.to(self.device)
            #             model_kwargs = dict(cond=cond)
            #             pred_x = self.diffusion.ddim_sample_loop(
            #                 self.model, 
            #                 x.shape, 
            #                 noise = x_T,
            #                 model_kwargs = model_kwargs,
            #                 clip_denoised = True
            #             )
            #             # SSIMスコアの計算（SSIM損失は1からSSIMインデックスを引いたもの）
            #             ssim = tgm.losses.SSIM(5, reduction='mean')
            #             SSIM_loss = ssim(x, pred_x)               
            #             test_SSIM_list.append(SSIM_loss.item())
            #         test_average_SSIM = sum(test_SSIM_list) / len(test_SSIM_list)

            #         if test_average_SSIM < self.best_score:
            #             self.best_score = test_average_SSIM
            #             save_model(self.model, 'best', self.dir_path)

                    

                # if self.wandb_flag:
                #     wandb_image=[wandb.Image(image[i]) for i in range(len(image))]
                #     wandb_x    =[wandb.Image(x[i]) for i in range(len(x))]
                #     wandb_pred_x    =[wandb.Image(pred_x[i]) for i in range(len(pred_x))]  
                #     wandb_cond=[wandb.Image(cond[i]) for i in range(len(cond))]


                #     wandb.log({
                #         'train_loss':train_average_loss,
                #         'test_score':test_average_SSIM,
                #         'image':wandb_image,
                #         'x':wandb_x,
                #         'pred_x':wandb_pred_x,
                #         'cond':wandb_cond,
                #     })

            import seaborn as sns
            from torch.nn.functional import interpolate
            cond_feature_list = []
            map = []
            mean_cond = torch.mean(cond_feature, dim=1, keepdim=True)
            for i in range(len(cond)):
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                sns.heatmap(mean_cond[i][0].cpu().numpy(), cmap="viridis", ax=ax, cbar=True)
                ax.axis('off')
                cond_feature_list.append(fig)
                
                resized_feature = interpolate(mean_cond, size=(256, 256), mode='bilinear', align_corners=False)

                fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
                # 背景に元画像を表示
                ax2.imshow(cond[i, 0].cpu().numpy(), cmap='gray', interpolation='none')

                # フィーチャーマップを重ね合わせ
                ax2.imshow(resized_feature[i, 0].cpu().numpy(), cmap='viridis', alpha=0.9, interpolation='none')
                # 軸を非表示
                ax2.axis('off')
                map.append(fig2)

            if True:
                if self.wandb_flag:
                    wandb_image=[wandb.Image(image[i]) for i in range(len(image))]
                    wandb_x    =[wandb.Image(x[i]) for i in range(len(x))]
                    wandb_pred_x    =[wandb.Image(pred_x[i]) for i in range(len(pred_x))]  
                    wandb_cond=[wandb.Image(cond[i]) for i in range(len(cond))]
                    wandb_feature_cond = [wandb.Image(cond_feature_list[i]) for i in range(len(cond_feature_list))]
                    wandb_map = [wandb.Image(map[i]) for i in range(len(map))]


                    wandb.log({
                        "feature_cond": wandb_feature_cond,
                        "map": wandb_map, 
                        'train_loss':train_average_loss,
                        'image':wandb_image,
                        'x':wandb_x,
                        'pred_x':wandb_pred_x,
                        'cond':wandb_cond,
                    })
        # save_model(self.model,'last', self.dir_path)
        wandb.finish()