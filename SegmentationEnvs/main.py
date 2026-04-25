# from Part1_position_model import UNet_128_2D, UNet_64_2D
# from Part2_segmentation_model.patchsize_128_2D import PatchSegUnet
from onnx import load_model
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import argparse
# from train_util import Trainer
from torch.optim import Adam

# from guided_diffusion.save_log import create_folder
# from guided_diffusion.script_util import create_gaussian_diffusion
# from guided_diffusion.save_log import load_model

from torch.optim import AdamW
import yaml
import numpy as np
import os

from utils import set_seed, get_dataset, get_vae, get_model, create_folder, select_type
from configs.config import DATASET_CONFIG_MAP, MODEL_CONFIG_MAP
from train_util import Trainer

def main(cfg):
    set_seed(cfg['seed'])        # シードを固定
    model = get_model(cfg) # モデルの作成
    run_dir = create_folder(cfg, cfg["run_name"], cfg["config_path"])    # フォルダの作成
    
    # 潜在空間を使うか
    if cfg["space"] == "pixel":
        vae = None
        del cfg["vae"]
    elif cfg["space"] == "latent":
        vae = get_vae(cfg)
           
    # 拡散モデルの選択
    if cfg["diffuser_type"]=="None" or cfg["diffuser_type"] is None:
        print("No Diffusion Model")
        diffuser = None
        criterion = nn.MSELoss()  # 拡散モデルを使用しない場合の損失関数
        del cfg["diffusion"]
        
    elif cfg["diffuser_type"]=="diffusion":
        print("Diffusion Model")
        diffuser  = select_type(cfg["diffuser_type"], cfg) # 拡散モデルの選択
        criterion = None  # 拡散モデルが内部で損失関数を持つ場合はNoneにする
        del cfg["diffusion"]["rectified_flow"]
    
    elif cfg["diffuser_type"]=="rectified_flow":
        print("Rectified Flow")
        diffuser  = select_type(cfg["diffuser_type"], cfg) # 拡散モデルの選択
        criterion = None  # 拡散モデルが内部で損失関数を持つ場合はNoneにする
        del cfg["diffusion"]["ddpm"]
    
    # データセットの取得
    train_set, val_set, test_set = get_dataset(cfg)
    cfg["data"]["train_size"]  = len(train_set)
    cfg["data"]["val_size"]    = len(val_set)
    cfg["data"]["test_size"]   = len(test_set)
    print(f"train size: {cfg['data']['train_size']}, val size: {cfg['data']['val_size']}, test size: {cfg['data']['test_size']}")

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"])     # 最適化手法

    para            = sum([np.prod(list(p.size())) for p in model.parameters()])
    trainable_param = sum(param.numel() for param in model.parameters() if param.requires_grad)
    cfg["model"]["num_param"] = para
    cfg["model"]["num_trainable_param"] = trainable_param
    print(f"Model {cfg['model']['model_name']} has {para} parameters, trainable parameters: {trainable_param}")
    print(f"Model {cfg['model']['model_name']} has {para/1e6:.2f}M parameters, trainable parameters: {trainable_param/1e6:.2f}M")
    
    trainer = Trainer(
        model       = model,
        diffuser    = diffuser,
        optimizer   = optimizer,
        train_set   = train_set,
        val_set     = val_set,
        test_set    = test_set,
        cfg         = cfg,
        dir_path    = run_dir,
        vae         = vae if cfg["space"] == "latent" else None,
        pos_encoder = None, # 今回は位置エンコーダーは使わない
        criterion   = criterion, # 拡散モデル以外で使用
    )
    
    trainer.train_loop(cfg)
    if cfg["task"]=="seg":
        trainer.test_loop(cfg)
    

    # if args.task_select == 'part1':        
    #     from wddd2_dataset_pos import WDDD2Dataset
    #     train_set   = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftrain.csv', transform=transform)
    #     test_set    = WDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Ftest.csv', transform=val_transform)
    #     val_set     = test_set
    #     if args.crop_size == 128:
    #         UNet=UNet_128_2D
    #     elif args.crop_size == 64:
    #         UNet=UNet_64_2D
            
    #     model = UNet(in_ch=args.in_channels, out_ch=args.out_channels, cond_in_ch=2 ,hidden_ch=32, time_embed_dim=100)
    #     pos_encoder = None

    # elif args.task_select == 'part2':
    #     from wddd2_dataset_Seg import SegWDDD2Dataset
    #     train_set = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segtrain.csv', transform=transform)
    #     val_set   = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segval.csv', transform=val_transform)
    #     test_set  = SegWDDD2Dataset(csv_path='/root/save/dataset/WDDD2_csv/Segtest.csv', transform=val_transform)

    #     part1_model = UNet(in_ch=1, out_ch=1, cond_in_ch=2 ,hidden_ch=32, time_embed_dim=100)
    #     part1_model = load_model(part1_model, args.part1_path)
    #     pos_encoder = part1_model.position_encoder
    #     model = PatchSegUnet(in_ch=args.in_channels, out_ch=args.out_channels, cond_in_ch=1 ,hidden_ch=32, time_embed_dim=100)



    # criterion = torch.nn.MSELoss()
    # trainer = Trainer(
    #     model=model,
    #     pos_encoder = pos_encoder,
    #     diffusion = diffusion,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     train_set=train_set,
    #     val_set=val_set,
    #     test_set=test_set,
    #     args=args,
    #     dir_path=dir_path,
    # )
    
    # trainer.train()
    # trainer.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    # ── yaml読み込み ──────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = args.config  # configのパスをcfgに保存（create_folderで使用）
    # dataset_configを読み込み＋cfgに統合
    dataset_name = cfg["dataset"]
    cfg["data"]["dataset"] = dataset_name
    del cfg["dataset"]  # datasetはdataの中に入れる
    dataset_yaml_path = DATASET_CONFIG_MAP[dataset_name]
    with open(dataset_yaml_path) as f:
        dataset_cfg = yaml.safe_load(f)
    cfg["data"].update(dataset_cfg)
    
    # model_configを読み込み＋cfgに統合
    # model_config_path = cfg["model_config_path"]
    model_config_path = MODEL_CONFIG_MAP[cfg["model"]["model_name"]]
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    cfg["model"].update(model_cfg)

    main(cfg)
        
    