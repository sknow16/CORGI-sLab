from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import wandb

class Trainer:
    def __init__(
            self,
            *, # *以降は呼び出す際にキーワード引数で指定する必要がある
            model,
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
        
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=True)
        self.val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,drop_last=False)
        
        self.epochs = args.epochs

        self.wandb_flag = args.wandb_flag
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
            val_loss_list = []
            self.model.train()

            for batch in tqdm(self.train_loader):
                image = batch['image'].to(self.device)
                mask = batch['label'].to(self.device)
                pred_mask = self.model(image)
                self.optimizer.zero_grad()
                loss = self.criterion(pred_mask, mask)
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())

            self.model.eval()

            for batch in tqdm(self.val_loader):
                image = batch['image'].to(self.device)
                mask = batch['label'].to(self.device)
                with torch.no_grad():
                    pred_mask = self.model(image)
                    loss = self.criterion(pred_mask, mask)
                    val_loss_list.append(loss.item())

            train_average_loss = sum(train_loss_list)/len(train_loss_list)
            val_average_loss   = sum(val_loss_list)/len(val_loss_list)

            if self.wandb_flag:
                pred_mask = pred_mask.cpu().numpy()
                mask=mask.cpu().numpy()
                image = image.cpu().numpy()
                wandb_pred_mask = [wandb.Image(pred_mask[i,:,:,:,64]) for i in range(len(pred_mask))]
                wandb_mask = [wandb.Image(mask[i,:,:,:,64]) for i in range(len(mask))]
                wandb_image0 = [wandb.Image(image[i,0,:,:,64]) for i in range(len(image))]
                wandb_image1 = [wandb.Image(image[i,1,:,:,64]) for i in range(len(image))]
                wandb_image2 = [wandb.Image(image[i,2,:,:,64]) for i in range(len(image))]
                wandb_image3 = [wandb.Image(image[i,3,:,:,64]) for i in range(len(image))]

                wandb.log({
                    'train_loss':train_average_loss,
                    'val_loss':val_average_loss,
                    'pred_mask':wandb_pred_mask,
                    'mask':wandb_mask,
                    'image0':wandb_image0,
                    'image1':wandb_image1,
                    'image2':wandb_image2,
                    'image3':wandb_image3,
                })
        
        self.model.eval()
        test_loss_list = []
        for batch in tqdm(self.test_loader):
            image = batch['image'].to(self.device)
            mask = batch['label'].to(self.device)
            with torch.no_grad():
                pred_mask = self.model(image)
                loss = self.criterion(pred_mask, mask)
                test_loss_list.append(loss.item())
        test_average_loss   = sum(test_loss_list)/len(test_loss_list)

        if self.wandb_flag:
            pred_mask = pred_mask.cpu().numpy()
            mask=mask.cpu().numpy()
            image = image.cpu().numpy()
            wandb_pred_mask = [wandb.Image(pred_mask[i,:,:,:,64]) for i in range(len(pred_mask))]
            wandb_mask = [wandb.Image(mask[i,:,:,:,64]) for i in range(len(mask))]
            wandb_image0 = [wandb.Image(image[i,0,:,:,64]) for i in range(len(image))]
            wandb_image1 = [wandb.Image(image[i,1,:,:,64]) for i in range(len(image))]
            wandb_image2 = [wandb.Image(image[i,2,:,:,64]) for i in range(len(image))]
            wandb_image3 = [wandb.Image(image[i,3,:,:,64]) for i in range(len(image))]
            wandb.log({
                    'test_loss':test_average_loss,
                    'pred_mask':wandb_pred_mask,
                    'mask':wandb_mask,
                    'image0':wandb_image0,
                    'image1':wandb_image1,
                    'image2':wandb_image2,
                    'image3':wandb_image3,
                })
        wandb.finish()