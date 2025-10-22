import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset



class brain_dataset(Dataset):
    def __init__(self,df,transform=None):
        self.id_list=df.iloc[:,0].tolist()
        self.transform=transform
        self.img_dir="/root/save/dataset/Task01_BrainTumour/imagesTr"
        self.mask_dir="/root/save/dataset/Task01_BrainTumour/labelsTr"
    
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, index):
        id=self.id_list[index]
        img_path=os.path.join(self.img_dir,id)
        mask_path=os.path.join(self.mask_dir,id)
       # Load image and mask
        img = nib.load(img_path).get_fdata()   # (H, W, D, C)
        mask = nib.load(mask_path).get_fdata() # (H, W, D)
        img = np.transpose(img, (3, 0, 1, 2))  # (H, W, D, C) -> (C, H, W, D)
        # img = img[0]
        # img = np.expand_dims(img,axis=0)
        mask = np.expand_dims(mask, axis=0)    # (H, E, D) -> (C, H, W, D)
        # maskのラベルを3から4に変換
        np.place(mask, mask == 3, 4)

        sample = {'image': img, 'label': mask}

        if self.transform:
            sample = self.transform(sample)
        
        return sample