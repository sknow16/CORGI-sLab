from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from metric.metric import jaccard
import wandb
from skimage.filters import threshold_otsu

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
        self.test_size = args.test_size
        self.val_size  = args.val_size
        self.wandb_flag = args.wandb_flag
        if self.wandb_flag:
            wandb.init(
                project=args.project_name,
                tags=[args.dim_size],
                name=args.model_name,
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
            print(f"===epoch({epoch+1}/{self.epochs})===")
            train_loss_list = []
            val_loss_list = []
            jaccard_0_list = []
            jaccard_1_list = []
            jaccard_2_list = []
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
            train_average_loss = sum(train_loss_list)/len(train_loss_list)
            if (epoch+1)%10!=0:
                wandb.log({
                    'train_loss':train_average_loss,
                })
            else:
                self.model.eval()
                for batch in tqdm(self.val_loader):
                    image = batch['image'].to(self.device)
                    mask = batch['label'].to(self.device)
                    with torch.no_grad():
                        pred_mask = self.model(image)
                        loss = self.criterion(pred_mask, mask)
                        val_loss_list.append(loss.item())
                        pred_mask = torch.sigmoid(pred_mask)
                        pred_mask_0 = pred_mask[:,0,:,:,:]
                        pred_mask_1 = pred_mask[:,1,:,:,:]
                        pred_mask_2 = pred_mask[:,2,:,:,:]
                        th0 = threshold_otsu(pred_mask_0.cpu().numpy())
                        th1 = threshold_otsu(pred_mask_1.cpu().numpy())
                        th2 = threshold_otsu(pred_mask_2.cpu().numpy())
                        pred_mask_0_binary = (pred_mask_0>th0).float().cpu().numpy()
                        pred_mask_1_binary = (pred_mask_1>th1).float().cpu().numpy()
                        pred_mask_2_binary = (pred_mask_2>th2).float().cpu().numpy()
                        
                        mask_0 = mask[:,0,:,:,:].cpu().numpy()
                        mask_1 = mask[:,1,:,:,:].cpu().numpy()
                        mask_2 = mask[:,2,:,:,:].cpu().numpy()

                        # threshold = threshold_otsu(pred_mask.cpu().numpy())
                        # pred_mask_binary = (pred_mask > threshold).float()
                        # pred_mask_binary = pred_mask_binary.cpu().numpy()
                        # mask      = mask.cpu().numpy()
                    jacard_0_score = jaccard(pred_mask_0_binary, mask_0)
                    jacard_1_score = jaccard(pred_mask_1_binary, mask_1)
                    jacard_2_score = jaccard(pred_mask_2_binary, mask_2)
                    # jacard_score = jaccard(pred_mask_binary, mask)
                    # jaccard_list.append(jacard_score*len(mask))
                    jaccard_0_list.append(jacard_0_score*len(mask))
                    jaccard_1_list.append(jacard_1_score*len(mask))
                    jaccard_2_list.append(jacard_2_score*len(mask))

                # jaccard_average = sum(jaccard_list)/self.val_size
                jaccard_0_average = sum(jaccard_0_list)/self.val_size
                jaccard_1_average = sum(jaccard_1_list)/self.val_size
                jaccard_2_average = sum(jaccard_2_list)/self.val_size
                miou              = (jaccard_0_average+jaccard_1_average+jaccard_2_average)/3
                val_average_loss   = sum(val_loss_list)/len(val_loss_list)

                if self.wandb_flag:
                    pred_mask = pred_mask.cpu().numpy()
                    image = image.cpu().numpy()
                    wandb_pred_mask0 = [wandb.Image(pred_mask[i,0,:,:,32]) for i in range(len(pred_mask))]
                    wandb_pred_mask1 = [wandb.Image(pred_mask[i,1,:,:,32]) for i in range(len(pred_mask))]
                    wandb_pred_mask2 = [wandb.Image(pred_mask[i,2,:,:,32]) for i in range(len(pred_mask))]
                    wandb_pred_mask_binary0 = [wandb.Image(pred_mask_0_binary[i,:,:,32]) for i in range(len(pred_mask))]
                    wandb_pred_mask_binary1 = [wandb.Image(pred_mask_1_binary[i,:,:,32]) for i in range(len(pred_mask))]
                    wandb_pred_mask_binary2 = [wandb.Image(pred_mask_2_binary[i,:,:,32]) for i in range(len(pred_mask))]

                    wandb_mask0 = [wandb.Image(mask_0[i,:,:,32]) for i in range(len(mask))]
                    wandb_mask1 = [wandb.Image(mask_1[i,:,:,32]) for i in range(len(mask))]
                    wandb_mask2 = [wandb.Image(mask_2[i,:,:,32]) for i in range(len(mask))]

                    wandb_image0 = [wandb.Image(image[i,0,:,:,32]) for i in range(len(image))]
                    wandb_image1 = [wandb.Image(image[i,1,:,:,32]) for i in range(len(image))]
                    wandb_image2 = [wandb.Image(image[i,2,:,:,32]) for i in range(len(image))]
                    wandb_image3 = [wandb.Image(image[i,3,:,:,32]) for i in range(len(image))]

                    wandb.log({
                        'train_loss':train_average_loss,
                        'val_loss':val_average_loss,
                        'pred_mask0':wandb_pred_mask0,
                        'pred_mask1':wandb_pred_mask1,
                        'pred_mask2':wandb_pred_mask2,
                        'pred_mask_binary0':wandb_pred_mask_binary0,
                        'pred_mask_binary1':wandb_pred_mask_binary1,
                        'pred_mask_binary2':wandb_pred_mask_binary2,
                        'mask0':wandb_mask0,
                        'mask1':wandb_mask1,
                        'mask2':wandb_mask2,
                        'IoU_ch0' :jaccard_0_average,
                        'IoU_ch1' :jaccard_1_average,
                        'IoU_ch2' :jaccard_2_average,
                        'mIoU'    :miou,
                        'image0':wandb_image0,
                        'image1':wandb_image1,
                        'image2':wandb_image2,
                        'image3':wandb_image3,
                    })
        
        self.model.eval()
        test_loss_list = []
        jaccard_0_list = []
        jaccard_1_list = []
        jaccard_2_list = []

        for batch in tqdm(self.test_loader):
            image = batch['image'].to(self.device)
            mask = batch['label'].to(self.device)
            with torch.no_grad():
                pred_mask = self.model(image)
                loss = self.criterion(pred_mask, mask)
                test_loss_list.append(loss.item())
                pred_mask_0 = pred_mask[:,0,:,:,:]
                pred_mask_1 = pred_mask[:,1,:,:,:]
                pred_mask_2 = pred_mask[:,2,:,:,:]
                th0 = threshold_otsu(pred_mask_0.cpu().numpy())
                th1 = threshold_otsu(pred_mask_1.cpu().numpy())
                th2 = threshold_otsu(pred_mask_2.cpu().numpy())
                pred_mask_0_binary = (pred_mask_0>th0).float().cpu().numpy()                    
                pred_mask_1_binary = (pred_mask_1>th1).float().cpu().numpy()
                pred_mask_2_binary = (pred_mask_2>th2).float().cpu().numpy()
                    
                mask_0 = mask[:,0,:,:,:].cpu().numpy()
                mask_1 = mask[:,1,:,:,:].cpu().numpy()
                mask_2 = mask[:,2,:,:,:].cpu().numpy()

                # pred_mask = torch.sigmoid(pred_mask)
                # threshold = threshold_otsu(pred_mask.cpu().numpy())
                # pred_mask_binary = (pred_mask > threshold).float()
                # pred_mask_binary = pred_mask_binary.cpu().numpy()
                # mask      = mask.cpu().numpy()
            jacard_0_score = jaccard(pred_mask_0_binary, mask_0)
            jacard_1_score = jaccard(pred_mask_1_binary, mask_1)
            jacard_2_score = jaccard(pred_mask_2_binary, mask_2)

            jaccard_0_list.append(jacard_0_score*len(mask))
            jaccard_1_list.append(jacard_1_score*len(mask))
            jaccard_2_list.append(jacard_2_score*len(mask))

        jaccard_0_average = sum(jaccard_0_list)/self.test_size
        jaccard_1_average = sum(jaccard_1_list)/self.test_size
        jaccard_2_average = sum(jaccard_2_list)/self.test_size
        miou              = (jaccard_0_average+jaccard_1_average+jaccard_2_average)/3
        test_average_loss   = sum(test_loss_list)/len(test_loss_list)

        if self.wandb_flag:
            pred_mask = pred_mask.cpu().numpy()
            image = image.cpu().numpy()
            wandb_pred_mask0 = [wandb.Image(pred_mask[i,0,:,:,32]) for i in range(len(pred_mask))]
            wandb_pred_mask1 = [wandb.Image(pred_mask[i,1,:,:,32]) for i in range(len(pred_mask))]
            wandb_pred_mask2 = [wandb.Image(pred_mask[i,2,:,:,32]) for i in range(len(pred_mask))]

            wandb_mask0 = [wandb.Image(mask_0[i,:,:,32]) for i in range(len(mask))]
            wandb_mask1 = [wandb.Image(mask_1[i,:,:,32]) for i in range(len(mask))]
            wandb_mask2 = [wandb.Image(mask_2[i,:,:,32]) for i in range(len(mask))]
            wandb_image0 = [wandb.Image(image[i,0,:,:,32]) for i in range(len(image))]
            wandb_image1 = [wandb.Image(image[i,1,:,:,32]) for i in range(len(image))]
            wandb_image2 = [wandb.Image(image[i,2,:,:,32]) for i in range(len(image))]
            wandb_image3 = [wandb.Image(image[i,3,:,:,32]) for i in range(len(image))]
            wandb_pred_mask_binary0 = [wandb.Image(pred_mask_0_binary[i,:,:,32]) for i in range(len(pred_mask_binary))]
            wandb_pred_mask_binary1 = [wandb.Image(pred_mask_1_binary[i,:,:,32]) for i in range(len(pred_mask_binary))]
            wandb_pred_mask_binary2 = [wandb.Image(pred_mask_2_binary[i,:,:,32]) for i in range(len(pred_mask_binary))]
            wandb.log({
                    'test_loss':test_average_loss,
                    'pred_mask0':wandb_pred_mask0,
                    'pred_mask1':wandb_pred_mask1,
                    'pred_mask2':wandb_pred_mask2,
                    'pred_mask_binary0':wandb_pred_mask_binary0,
                    'pred_mask_binary1':wandb_pred_mask_binary1,
                    'pred_mask_binary2':wandb_pred_mask_binary2,
                    'mask0':wandb_mask0,
                    'mask1':wandb_mask1,
                    'mask2':wandb_mask2,
                    'IoU_ch0' :jaccard_0_average,
                    'IoU_ch1' :jaccard_1_average,
                    'IoU_ch2' :jaccard_2_average,
                    'mIoU'    :miou,
                    'image0':wandb_image0,
                    'image1':wandb_image1,
                    'image2':wandb_image2,
                    'image3':wandb_image3,
                })
        wandb.finish()