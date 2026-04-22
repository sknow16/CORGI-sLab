import torch
import numpy as np
import random
import os
import shutil
from utils.config import DATASET_CONFIG_MAP, MODEL_CONFIG_MAP
def set_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # GPUの乱数も固定（必要なら）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # マルチGPU用
        torch.backends.cudnn.deterministic = True  # CuDNNの動作を固定
        torch.backends.cudnn.benchmark = False  # 再現性を優先

def create_folder(cfg, run_name, config_path):
    model_name = cfg["model"]["model_name"]
    dataset    = cfg["data"]["dataset"]
    task       = cfg["task"]
    task_str   = f"({task})" if task else ""

    base_dir = os.path.join(cfg["path"], f"{model_name}{task_str}:{dataset}")
    run_dir  = os.path.join(base_dir, run_name)

    # ── 既存チェック ──────────────────────────────
    if os.path.exists(run_dir):
        print(f"[WARNING] 既に同じrun_nameのフォルダが存在します: {run_dir}")
        answer = input("上書きしますか？ [y/n]: ")
        if answer.lower() != "y":
            raise RuntimeError(f"中断しました。run_nameを変更してください: {run_name}")
        
    os.makedirs(os.path.join(run_dir, "imgs"),    exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)

    # ── config類をまとめて保存 ────────────────────
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))

    dataset_config_path = DATASET_CONFIG_MAP[cfg["data"]["dataset"]]
    shutil.copy(dataset_config_path, os.path.join(run_dir, "dataset_config.yaml"))

    model_config_path = MODEL_CONFIG_MAP[cfg["model"]["model_name"]]
    shutil.copy(model_config_path, os.path.join(run_dir, "model_config.yaml"))

    return run_dir