import torch
import monai

def get_model(cfg):
    
    if cfg["model"]["model_name"] == "PatchPosUnet" and cfg["task"] == "pos":
        from models import PatchPosUnet
        model = PatchPosUnet(cfg["crop_size"],cfg["model"]["model_config"])
    
    elif cfg["model"]["model_name"] == "PatchSegUnet" and cfg["task"] == "seg":
        #  from models.Part2_segmentation_model import PatchSegUnet
        pass
    
    elif cfg["model"]["model_name"] == "LSegDiff":
        from models import LSegUNet
        model = LSegUNet(**cfg["model"]["model_config"])
        
        
    elif cfg["model"]["model_name"] == "UNet":
        from monai.networks.nets import UNet
        model = UNet(**cfg["model"]["model_config"])
    elif cfg["model"]["model_name"] == "swinunetr":
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            in_channels=cfg["model"]["img_channels"],
            out_channels=cfg["model"]["mask_channels"],
            feature_size=24,       # 12の倍数であること（ソース内でチェックあり）
            spatial_dims=2,
            patch_size=2,          # デフォルト2。入力は 2**5=32 の倍数が必要
            use_checkpoint=True,
        )
    else:
        raise NotImplementedError(f"Model {cfg['model']['model_name']} is not implemented.")
    return model