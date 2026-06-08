from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from utils.utils import load_model, requires_grad

# from utils.lora import insert_lora_adapters

def get_vae(cfg):
    # 学習済みモデル
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    in_out_channel = cfg["vae"]["in_out_ch"]
        
    if in_out_channel != 3:
        # エンコーダ部の調整
        vae.encoder.conv_in = nn.Conv2d(in_out_channel, 128, kernel_size=3, stride=1, padding=1)
        # デコーダ部の調整
        vae.decoder.conv_out = nn.Conv2d(128, in_out_channel, kernel_size=3, stride=1, padding=1)
    # # LoRAの挿入
    # if cfg["vae"]["mode"] == "adapter":
    #     vae = insert_lora_adapters(vae, rank=cfg["vae"]["lora_rank"], alpha=cfg["vae"]["lora_alpha"])
    # モデル全体のパラメータをフリーズ
    for param in vae.parameters():
        param.requires_grad = False
    
    if cfg["vae"]["checkpoint"]:
        print("Reading the VAE checkpoint...")
        print(cfg["vae"]["path"])
        vae = load_model(vae, cfg["vae"]["path"])
    requires_grad(vae, False)
    return vae