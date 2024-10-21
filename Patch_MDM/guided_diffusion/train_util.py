from torch.utils.data import DataLoader
import torch as th
from tqdm.auto import tqdm
# from metric.metric import dice, jaccard
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
        self.device = th.device("cuda0" if th.cuda.is_available() else "cpu")
        print(self.device)
        self.model = model.to(self.device)
        self.diffusion    = diffusion

        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,drop_last=True)
        self.val_loader   = DataLoader(val_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,drop_last=False)
        # self.test_loader  = DataLoader(test_set, shuffle=False,batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,drop_last=False)

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
        # self.test_volume = args.test_volume

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
                # "test_volume": args.test_volume
                }
            )

    def train(self):
        for epoch in range(1,self.epochs+1):
            print(f"epoch:{epoch}/{self.epochs}")
            train_losses = []
            val_losses   = []

            # 学習
            self.model.train()
            # for image, mask in tqdm(self.train_loader):
            for image in tqdm(self.train_loader):
                x_start = image.to(self.device)  # mask画像            
                t = th.randint(0, self.diffusion.num_timesteps, (x_start.shape[0],), device=self.device)
                loss_dict = self.diffusion.training_losses(self.model, x_start, t)
                loss = loss_dict["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

             
                        
            # サンプリング (val)
            # 今は固定で２５エポックごとにサンプリングして評価
            self.model.eval()
            if epoch % 25 == 0:
                for image in self.val_loader:
                    x_start = image.to(self.device)
                    pred_x_start_list = []
                    with th.no_grad:
                        t = th.randint(0, self.diffusion.num_timesteps, (x_start.shape[0],), device=self.device)
                        loss_dict = self.diffusion.training_losses(self.model, x_start, t)
                        loss = loss_dict["loss"]
                        val_losses.append(loss.item())
                    x_start = x_start.cpu().numpy()

            # 25エポック毎で精度を計算し、それ以外は１バッチ分だけサンプリング
            else:
                for image in self.val_loader:
                    x_start = image.to(self.device)
                    self.model.eval()
                    with th.no_grad():
                        t = th.tensor([250])
                        x_t = self.diffusion.q_sample(x_start, t)
                        pred_x_start = self.model(x_t, t)
                    break

            trian_avg_loss = sum(train_losses)/len(train_losses)
            print(f"Epoch {epoch} finished.")
            print(f"train_loss:{trian_avg_loss}")


            if self.wandb_flag:
                wandb_num_images = self.wandb_num_images if self.wandb_num_images<=x_start.shape[0] else x_start.shape[0]
                if epoch % 25 == 0: # 今は固定で２５エポックごとにサンプリングして評価
                    wandb.log({
                        "train_loss":trian_avg_loss,
                        "val_loss": sum(val_losses)/len(val_losses)
                        })

                else:
                    wandb_image  = [wandb.Image(x_start[i]) for i in range(wandb_num_images)]
                    wandb_pred_x = [wandb.Image(pred_x_start[i]) for i in range(wandb_num_images)]
                    wandb_x_t    = [wandb.Image(x_t[i]) for i in range(wandb_num_images)]
                    wandb.log({
                        "train_loss":trian_avg_loss,
                        "x_0":wandb_image,
                        "x_t":wandb_x_t,
                        "pred_x_0":wandb_pred_x,
                        })
    
            # if (epoch)%self.save_n_model == 0:
            #     save_model(self.model,epoch,self.dir_path)
        if self.wandb_flag:
            wandb.finish()
