import glob
import re
import torch as th
from collections import OrderedDict
from metric.metric import dice, jaccard, sensitivity, specificity, accuracy, hausdorff_distance, hausdorff_distance_95
import scipy.stats as stats
import numpy as np
import os

eps = 1e-3
@th.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)
    
def sample_t(num_samples, exponential_pdf = ExponentialPDF(a=0, b=1, name='ExponentialPDF'), a=4):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = th.from_numpy(t).float()
    t = th.cat([t, 1 - t], dim=0)
    t = t[th.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def logit_normal_sample(batch_size, mean=0.0, std=1.0):
    # 標準正規分布からサンプリング
    u = th.randn(batch_size) * std + mean
    # シグモイド関数を適用
    t = th.sigmoid(u)
    return t

def calc_metric(pred_x, x):
    dice_list = []
    iou_list = []
    sensitivity_list = []
    specificity_list = []
    accuracy_list = []
    hausdorff_distance_list = []
    hausdorff_distance_95_list = []
    batch = x.shape[0]
    for i in range(batch):
        target = x[i]
        pred = pred_x[i]
        dice_list.append(dice(pred, target))
        iou_list.append(jaccard(pred, target))
        sensitivity_list.append(sensitivity(pred, target))
        specificity_list.append(specificity(pred, target))
        accuracy_list.append(accuracy(pred, target))
        hausdorff_distance_list.append(hausdorff_distance(pred, target))
        hausdorff_distance_95_list.append(hausdorff_distance_95(pred, target))
        
    metric = {
        "dice":         sum(dice_list),
        "iou":          sum(iou_list),
        "specificity":  sum(specificity_list),
        "sensitivity":  sum(sensitivity_list),
        "accuracy":     sum(accuracy_list),
        "hausdorff":    sum(hausdorff_distance_list),
        "hausdorff_95": sum(hausdorff_distance_95_list),
    }

    return metric

def save_model(model, epoch, dir_path, optimizer=None, ema=None, save_n=2):
    """
    in
    ===============================
    model: モデルの重み
    epoch: 現在のエポック数
    dir_path: ディレクトリを指定
    optimizer: オプティマイザ（任意）
    ema: EMAモデル（任意）
    save_n: 保存するチェックポイントの数

    out
    ===============================
    重み保存 + 古いcheckpointを削除
    """
    model_path = os.path.join(dir_path, "weights", f"weight_epoch_{epoch}.pth")
    th.save(model.state_dict(), model_path)

    if optimizer is not None:
        optimizer_path = os.path.join(dir_path, "weights", f"optimizer_epoch_{epoch}.pth")
        th.save(optimizer.state_dict(), optimizer_path)

    if ema is not None:
        ema_path = os.path.join(dir_path, "weights", f"weight_epoch_{epoch}_ema.pth")
        th.save(ema.state_dict(), ema_path)

    # modelのcheckpoint削除
    checkpoints = sorted([
        f for f in glob.glob(os.path.join(dir_path, "weights", "weight_epoch_*.pth"))
        if re.search(r"weight_epoch_\d+\.pth$", f)
    ])
    if len(checkpoints) > save_n:
        for old_ckpt in checkpoints[:-save_n]:
            os.remove(old_ckpt)
            print(f"削除: {os.path.basename(old_ckpt)}")

            # optimizerの削除
            old_opt = old_ckpt.replace("weight_epoch_", "optimizer_epoch_")
            if os.path.exists(old_opt):
                os.remove(old_opt)
                print(f"削除: {os.path.basename(old_opt)}")

            # EMAの削除（old_ckptのepoch番号を使って対応するEMAを削除）
            old_ema = old_ckpt.replace(".pth", "_ema.pth")
            if os.path.exists(old_ema):
                os.remove(old_ema)
                print(f"削除: {os.path.basename(old_ema)}")

    # EMAのcheckpointも削除
    ema_checkpoints = sorted([
        f for f in glob.glob(os.path.join(dir_path, "weights", "weight_epoch_*_ema.pth"))
        if re.search(r"weight_epoch_\d+_ema\.pth$", f)
    ])
    if len(ema_checkpoints) > save_n:
        for old_ema in ema_checkpoints[:-save_n]:
            os.remove(old_ema)
            print(f"削除: {os.path.basename(old_ema)}")
# def save_model(model,epoch,dir_path, optimizer=None, save_n=2):
#     """
#     in
#     ===============================
#     model: モデルの重み
#     epoch: 現在のエポック数
#     dir_path: ディレクトリを指定
#     save_n: 保存するチェックポイントの数
    
#     out
#     ===============================
#     重み保存
#     """
#     model_path = os.path.join(dir_path,"weights",f"weight_epoch_{epoch}.pth")
#     th.save(model.state_dict(), model_path)
#     if optimizer is not None:
#         optimizer_path = os.path.join(dir_path,"weights",f"optimizer_epoch_{epoch}.pth")
#         th.save(optimizer.state_dict(), optimizer_path)

#     checkpoints = sorted([
#         f for f in glob.glob(os.path.join(dir_path, "weights", "weight_epoch_*.pth"))
#         if re.search(r"weight_epoch_\d+\.pth$", f)
#     ])
#     if len(checkpoints) > save_n:
#         for old_ckpt in checkpoints[:-save_n]:
#             os.remove(old_ckpt)
#             print(f"削除: {os.path.basename(old_ckpt)}")
            
#             # 対応するoptimizerも削除
#             old_opt = old_ckpt.replace("weight_epoch_", "optimizer_epoch_")
#             if os.path.exists(old_opt):
#                 os.remove(old_opt)
#                 print(f"削除: {os.path.basename(old_opt)}")

def load_model(model,model_path):
    """
    in
    ===============================
    model: モデルの重み
    mode_path: モデルのパス
    """
    state_dict = th.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # if 'module.' in k: # 修正前
        if k.startswith('module.'): # 修正後
            new_state_dict[k[7:]] = v  # 'module.' を取り除いて新しい辞書に追加
        else:
            new_state_dict[k] = v  # 'module.' が含まれていない場合、そのまま新しい辞書に追加
            
    model.load_state_dict(new_state_dict)
    return model