from torch.utils.data import DataLoader
import torch as th
from tqdm.auto import tqdm

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
        test_set, # 今は使ってない
        args,
        dir_path,
        scheduler=None,
    ):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.diffusion    = diffusion
        self.train_loader = self.get_loader(dataset=train_set,shuffle=True,batch_size=args.batch_size,num_workers=args.num_workers)
        self.val_loader   = self.get_loader(dataset=val_set,shuffle=False,batch_size=args.batch_size,num_workers=args.num_workers)
        self.test_loader  = self.get_loader(dataset=val_set,shuffle=True,batch_size=args.batch_size,num_workers=args.num_workers)
        
        self.img_size   = args.img_size
        self.in_channels= args.in_channels
        
        self.optimizer  = optimizer
        self.lr         = args.lr
        self.scheduler  = scheduler # まだ使ってない
        self.epochs     = args.epochs
        
        self.wandb_flag = args.wandb_flag
        self.wandb_num_images = args.wandb_num_images
        self.save_n_model = args.save_n_model
        self.dir_path     = dir_path
        if self.wandb_flag:
            wandb.init(
                project=args.project_name,
                config={
                "model":         args.model_name,
                "dataset":       args.dataset,
                "epochs":        args.epochs,
                "image_size":    args.img_size,
                "channel":       args.in_channels,
                "batch_size":    args.batch_size,
                "learning_rate": args.lr,
                }
            )

    def train(self):
        for epoch in range(1,self.epochs+1):
            print(f"epoch:{epoch}/{self.epochs}")
            train_losses = []
            val_losses   = []

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
                    
            # 検証
            self.model.eval()
            for image, mask in tqdm(self.val_loader):
                x_start = mask.to(self.device)
                y       = image.to(self.device)
                model_kwargs = dict(conditioned_image=y)
                
                with th.no_grad():
                    t = th.randint(0, self.diffusion.num_timesteps, (x_start.shape[0],), device=self.device)                  
                    loss_dict = self.diffusion.training_losses(self.model, x_start, t, model_kwargs)
                    loss = loss_dict["loss"]
                    val_losses.append(loss.item())
                        
            # サンプリング (改良が必要)
            for image, mask in self.test_loader:
                x_start = mask.to(self.device)
                y       = image.to(self.device)
                break

            model_kwargs = dict(conditioned_image=y)
            x_end = th.randn(x_start.shape[0],self.in_channels,self.img_size,self.img_size).to(self.device)
            pred_x_start = self.diffusion.ddim_sample_loop(
                self.model,
                x_end.shape,
                model_kwargs=model_kwargs,
                clip_denoised=True,
            )
            
            trian_avg_loss = sum(train_losses)/len(train_losses)
            val_avg_loss   = sum(val_losses)/len(val_losses)
            print(f"Epoch {epoch} finished.")
            print(f"train_loss:{trian_avg_loss}|| val_loss:{val_avg_loss}")

            if self.wandb_flag:
                wandb_num_images = self.wandb_num_images if self.wandb_num_images<=x_start.shape[0] else x_start.shape[0]
                wandb_image = [wandb.Image(y[i]) for i in range(wandb_num_images)]
                wandb_mask  = [wandb.Image(x_start[i]) for i in range(wandb_num_images)]
                wandb_pred_mask = [wandb.Image(pred_x_start[i]) for i in range(wandb_num_images)]
                wandb.log({
                    "train_loss":trian_avg_loss,
                    "val_loss":val_avg_loss,
                    "image":wandb_image,
                    "mask":wandb_mask,
                    "pred_x":wandb_pred_mask
                    })
                
            if (epoch)%self.save_n_model == 0:
                save_model(self.model,epoch,self.dir_path)
        if self.wandb_flag:
            wandb.finish()

        
    def get_loader(self,dataset,shuffle,batch_size,num_workers):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True, # CPUからGPUにデータを転送する際にメモリをピン留め（これにより高速化）
            drop_last=True   # データローダ―の最後のバッチが他のバッチのサンプル数と異なる場合そのバッチは破棄 (実験の均一性を保つため)
        )
        return dataloader