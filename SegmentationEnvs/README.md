# 実験手順ガイド

## 目次
1. [既存モデルで実験設定を変える場合](#1-既存モデルで実験設定を変える場合)
2. [新しいモデルを追加する場合](#2-新しいモデルを追加する場合)
3. [新しいデータセットを追加する場合](#3-新しいデータセットを追加する場合)
4. [実行](#4-実行)

---

## 1. 既存モデルで実験設定を変える場合

### 1-1. `config.yaml` の修正

まず `configs/config.yaml` を開いて以下を確認・修正する。

#### 基本設定
```yaml
run_name: "モデルの詳細な名前"        # W&Bのrun名及び保存される名前。わかりやすい名前をつける(ex: unet_hidden32, ...)
model_name: "LSegDiff"   # 使用するモデル名
dataset: "WDDD2"         # 使用するデータセット名
task: seg                # pos or seg
diffuser_type: null      # null / diffusion / rectified_flow / flow_matching (nullの時は拡散ではない)
space: latent            # latent or pixel
crop_size: null          # null / 32 / 64 / 128（パッチ分割する場合）
wandb:
  project_name: "Wandbのプロジェクト名"
  ...
```


#### 学習パラメータ
```yaml
train:
  epochs: 1000           # 拡散系統は1000程度、他は500?
  lr: 1.0e-4
  batch_size(global): 16
  batch_size(micro): 16
  use_ema: true          # 指数移動平均モデルをしようするか(拡散モデルでは安定するため使用することが推奨)
  val_step_num: 50       # 何epochごとにvalidationするか
  save_n_model: 5        # 何epochごとにcheckpointを保存するか
  optimizer: "AdamW"
  loss_function: "MSELoss"
```

#### チェックポイントから再開する場合
```yaml
checkpoint:
  use_checkpoint: true
  checkpoint_epoch: 100  # 再開するepoch
  path: "/root/save/log2026/モデル名:データセット名/run名/weights/weight_epoch_100.pth"
```

---

### 1-2. `model_config.yaml` の修正（モデルのハイパラを変える場合）

`configs/models/モデル名/model_config.yaml` を開いて修正する。

#### 例: UNetの場合
```yaml
spatial_dims: 2
channels: [32, 64, 128, 256]
strides: [2, 2, 2]
num_res_units: 2
norm: "batch"     # batch / instance / group
act: "relu"       # relu / leakyrelu / prelu
dropout: 0.1
```


### 1-3. `dataset_config.yaml` の修正（データセット設定を変える場合）

`configs/datasets/データセット名.yaml` を開いて修正する。

#### 例: WDDD2の場合, csvファイルの指定先を変える
```yaml
seg:
  train_csv_path: "/root/save/dataset/WDDD2_csv/Segtrain.csv"
  val_csv_path:   "/root/save/dataset/WDDD2_csv/Segval.csv"
  test_csv_path:  "/root/save/dataset/WDDD2_csv/Segtest.csv"

pos:
  train_csv_path: "/root/save/dataset/WDDD2_csv/Postrain.csv"
  test_csv_path:  "/root/save/dataset/WDDD2_csv/Postest.csv"
```

---

## 2. 新しいモデルを追加する場合

### 手順

#### Step 1: モデルファイルを配置
```
models/
└── 新モデル名/
    |── new_model_ver1.py    ← モデルのクラスを実装
    └── new_model_ver2.py    
```

#### Step 2: `models/__init__.py` にimportを追加
```python
from .新モデル名.new_model_ver1 import NewModel
```

#### Step 3: `utils/get_model.py` に分岐を追加
```python
def get_model(cfg):
    model_name = cfg["model_name"]
    if model_name == "unet":
        ...
    elif model_name == "NewModel":   # ← 追加
        from models import NewModel
        model = NewModel(**cfg["model"]["model_config"])
```

#### Step 4: `configs/models/` にyamlを追加
```yaml
# configs/models/new_model.yaml
hidden_ch: 64
skip_flag: True # モデルを呼び出す際に必要なデフォルトパラメータを先にここに書いておく
# その他モデル固有のパラメータ
```

#### Step 5: `configs/config.py` の `MODEL_CONFIG_MAP` に追加
```python
MODEL_CONFIG_MAP = {
    "unet":     "configs/models/UNet/model_config.yaml",
    "NewModel": "configs/models/New_Model/model_config.yaml",  # ← 追加
}
```

#### Step 6: `_build_model_kwargs` に分岐が必要な場合
`train_util.py` の `_build_model_kwargs` に `kwargs_type` を追加する。

```python
elif kwargs_type == "new_kwargs":
    return dict(new_key=y)
```

---

## 3. 新しいデータセットを追加する場合

### 手順

#### Step 1: データセットファイルを配置
```
dataset/
└── 新データセット名/
    ├── new_dataset.py    ← Datasetクラスを実装
    └── new_dataset.yaml  ← データセット設定
```

#### Step 2: `dataset/__init__.py` にimportを追加
```python
from .新データセット名.new_dataset import NewDataset
```

#### Step 3: `utils/get_dataset.py` に分岐を追加
```python
def get_dataset(cfg):
    dataset = cfg["data"]["dataset"]
    if dataset == "WDDD2":
        ...
    elif dataset == "NewDataset":   # ← 追加
        from dataset import NewDataset
        train_set = NewDataset(csv_path=cfg["data"]["train_csv_path"], transform=transform)
        val_set   = NewDataset(csv_path=cfg["data"]["val_csv_path"],   transform=val_transform)
        test_set  = NewDataset(csv_path=cfg["data"]["test_csv_path"],  transform=val_transform)
    return train_set, val_set, test_set
```

#### Step 4: `configs/datasets/` にyamlを追加
```yaml
# configs/datasets/new_dataset.yaml
dataset: "NewDataset"
train_csv_path: "/path/to/train.csv"
val_csv_path:   "/path/to/val.csv"
test_csv_path:  "/path/to/test.csv"
```

#### Step 5: `utils/config.py` の `DATASET_CONFIG_MAP` に追加
```python
DATASET_CONFIG_MAP = {
    "WDDD2":      "dataset/wddd2/wddd2.yaml",
    "NewDataset": "dataset/新データセット名/new_dataset.yaml",  # ← 追加
}
```

#### Step 6: `val_loop` のbest score判定を確認
`train_util.py` の `val_loop_seg` でbest scoreの判定がチャネル数に対応しているか確認する。

```python
# mask_channels=1の場合
if self.mask_shape[0] == 1:
    current_score = avg_metrics["ch0_iou"]
# mask_channels=3の場合
elif self.mask_shape[0] == 3:
    current_score = (avg_metrics["ch1_iou"] + avg_metrics["ch2_iou"]) / 2
```

---

## 4. 実行

### 学習
```
cd SegmentationEnvs
python main.py
```

<!-- `run.sh` の内容：
```bash
#!/bin/bash
set -e
export WANDB_API_KEY="your_api_key"
python main.py --config configs/config.yaml
```

### テスト
```bash
# config.yamlのtest設定を確認してから実行
python main.py --config configs/config.yaml --mode test -->
```

### 結果確認
学習開始時にターミナルに表示されるW&BのURLからダッシュボードを確認する。

```
wandb: 🚀 View run at: https://wandb.ai/ユーザー名/プロジェクト名/runs/xxxxx
```

---

## チェックリスト

### 実験前
- [ ] `run_name` をわかりやすい名前に設定した
- [ ] `task`（pos/seg）を確認した
- [ ] `dataset` と `part` を確認した
- [ ] `diffuser_type` を確認した（null/diffusion/rectified_flow）
- [ ] `model_name` を確認した
- [ ] 学習パラメータ（lr, epochs, batch_size）を確認した
- [ ] データセットのcsvパスが正しいか確認した
- [ ] `train_size` / `val_size` / `test_size` が正しいか確認した（0の場合は自動取得）

### 新モデル追加時
- [ ] モデルファイルを `models/` に配置した
- [ ] `models/__init__.py` にimportを追加した
- [ ] `utils/get_model.py` に分岐を追加した
- [ ] `configs/models/` にyamlを追加した
- [ ] `utils/config.py` の `MODEL_CONFIG_MAP` に追加した

### 新データセット追加時
- [ ] データセットファイルを `dataset/` に配置した
- [ ] `dataset/__init__.py` にimportを追加した
- [ ] `utils/get_dataset.py` に分岐を追加した
- [ ] `configs/datasets/` にyamlを追加した
- [ ] `utils/config.py` の `DATASET_CONFIG_MAP` に追加した
- [ ] `val_loop` のbest score判定がチャネル数に対応しているか確認した