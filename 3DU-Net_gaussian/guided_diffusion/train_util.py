from torch.utils.data import DataLoader
import torch as th
from tqdm.auto import tqdm
from metric.metric import dice, jaccard
from guided_diffusion.save_log import save_model,save_imgs
import wandb

class Trainer:
    def __init__(
        self,
        *, # *以降は呼び出す際にキーワード引数で指定する必要がある(ex Trainer(a=1, b=2))
        model,
        diffusion,
        optimizer,
        train_set,
        val_set,
        test_set=None, # 今は使ってない
        args,
        dir_path,
        scheduler=None,
    ):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.diffusion    = diffusion

        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,drop_last=True)
        self.val_loader   = DataLoader(val_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,drop_last=False)
        self.test_loader  = DataLoader(test_set, shuffle=False,batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,drop_last=False)

        self.img_size   = args.image_size
        self.in_channels= args.in_channels
        
        self.optimizer  = optimizer
        self.lr         = args.lr
        self.scheduler  = scheduler # まだ使ってない
        self.epochs     = args.epochs
        
        self.wandb_flag = args.wandb_flag
        self.wandb_num_images = args.wandb_num_images
        self.save_n_model = args.save_n_model
        self.dir_path     = dir_path

        self.val_volume  = args.val_volume
        self.test_volume = args.test_volume

        if self.wandb_flag:
            wandb.init(
                project=args.project_name,
                tags=[args.dim_size],
                config={
                "model":         args.model_name,
                "dataset":       args.dataset,
                "epochs":        args.epochs,
                "image_size":    args.img_size,
                "channel":       args.in_channels,
                "batch_size":    args.batch_size,
                "learning_rate": args.lr,
                "train_volume": args.train_volume,
                "val_volume": args.val_volume,
                "test_volume": args.test_volume
                }
            )

    def train(self):
        for epoch in range(1,self.epochs+1):
            print(f"epoch:{epoch}/{self.epochs}")
            train_losses = []

            # 学習
            self.model.train()
            for image, mask in tqdm(self.train_loader):
                x_start = mask.to(self.device)  # mask画像
                y       = image.to(self.device) # MRI画像
                model_kwargs = dict(conditioned_image=y)
            
                t = th.randint(0, self.diffusion.num_timesteps, (x_start.shape[0],), device=self.device)
                loss_dict = self.diffusion.training_losses(self.model, x_start, t, model_kwargs)
                loss = loss_dict["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

             
                        
            # サンプリング (val)
            # 今は固定で２５エポックごとにサンプリングして評価
            self.model.eval()
            if epoch % 25 == 0:
                dice_list = []
                iou_list  = []
                for image, mask in self.val_loader:
                    x_start = mask.to(self.device)
                    y       = image.to(self.device)

                    model_kwargs = dict(conditioned_image=y)
                    pred_x_start_list = []

                    # アンサンブル数
                    # アンサンブル数もいまは固定で4
                    # 今後argsから指定したい
                    for _ in range(4):
                        with th.no_grad():
                            pred_x_start = self.diffusion.ddim_sample_loop(
                                self.model,
                                x_start.shape,
                                model_kwargs=model_kwargs,
                                clip_denoised=True,  # これはなに？　→　不明
                            )
                        pred_x_start = th.sigmoid(pred_x_start)
                        pred_x_start_list.append(pred_x_start)
                
                    mean_x_start = th.mean(th.stack(pred_x_start_list), dim=0)  # アンサンブルの平均
                    std_x_start = th.std(th.stack(pred_x_start_list), dim=0)    # アンサンブルの標準偏差
                    threshold = 0.6  # マスク画像生成のための閾値 # 今後、正しく設定する
                    mean_x_start_binary = (mean_x_start>threshold).cpu().numpy()
                    x_start = x_start.cpu().numpy()
                    dice_score = dice(mean_x_start_binary, x_start)
                    iou_score = jaccard(mean_x_start_binary, x_start)
                    # 
                    dice_list.append(dice_score*len(x_start))
                    iou_list.append(iou_score*len(x_start))
                dice_mean_score = sum(dice_list)/self.val_volume
                iou_mean_score  = sum(iou_list)/self.val_volume

            # 25エポック毎で精度を計算し、それ以外は１バッチ分だけサンプリング
            else:
                for image, mask in self.val_loader:
                    x_start = mask.to(self.device)
                    y       = image.to(self.device)

                    model_kwargs = dict(conditioned_image=y)

                    with th.no_grad():
                        pred_x_start = self.diffusion.ddim_sample_loop(
                            self.model,
                            x_start.shape,
                            model_kwargs=model_kwargs,
                            clip_denoised=True,  # これはなに？　→　不明      
                        )
                        pred_x_start = th.sigmoid(pred_x_start)
                    break

            trian_avg_loss = sum(train_losses)/len(train_losses)
            print(f"Epoch {epoch} finished.")
            print(f"train_loss:{trian_avg_loss}")


            if self.wandb_flag:
                wandb_num_images = self.wandb_num_images if self.wandb_num_images<=x_start.shape[0] else x_start.shape[0]
                wandb_image = [wandb.Image(y[i]) for i in range(wandb_num_images)]
                wandb_mask  = [wandb.Image(x_start[i]) for i in range(wandb_num_images)]

                if epoch % 25 == 0: # 今は固定で２５エポックごとにサンプリングして評価
                    wandb_pred_mask_binary = [wandb.Image(mean_x_start_binary[i]) for i in range(wandb_num_images)]
                    wandb.log({
                        "train_loss":trian_avg_loss,
                        "image":wandb_image,
                        "mask":wandb_mask,
                        "pred_mean_mask_binary":wandb_pred_mask_binary,
                        "dice":dice_mean_score,
                        "iou":iou_mean_score
                        })

                else:
                    wandb_pred_mask = [wandb.Image(pred_x_start[i]) for i in range(wandb_num_images)]
                    wandb.log({
                        "train_loss":trian_avg_loss,
                        "image":wandb_image,
                        "mask":wandb_mask,
                        "pred_mask":wandb_pred_mask,
                        })
    
            # if (epoch)%self.save_n_model == 0:
            #     save_model(self.model,epoch,self.dir_path)
        if self.wandb_flag:
            wandb.finish()
