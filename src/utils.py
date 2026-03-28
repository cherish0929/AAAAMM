import argparse
import json, os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import pyvista as pv


def set_seed(seed: int = 0):
    """设置随机种子，保证实验可重复。"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(m):
    """线性层和Attention层的简易初始化。"""
    if isinstance(m, nn.Linear):
        if m.weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None and m.bias.numel() > 0:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None and m.in_proj_bias.numel() > 0:
            m.in_proj_bias.data.fill_(0.01)
        if m.out_proj.weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None and m.out_proj.bias.numel() > 0:
            m.out_proj.bias.data.fill_(0.01)


class ChannelNormalizer:
    """通道级别的均值方差归一化工具。"""

    def __init__(self, mean, std, eps: float = 1e-6):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        self.eps = eps
        self.mean = torch.tensor(mean).view(1, 1, -1)
        self.std = torch.tensor(std).view(1, 1, -1)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean.to(tensor.device)) / (self.std.to(tensor.device) + self.eps)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std.to(tensor.device) + self.eps) + self.mean.to(tensor.device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def as_dict(self):
        return {
            "mean": self.mean.cpu().numpy().reshape(-1).tolist(),
            "std": self.std.cpu().numpy().reshape(-1).tolist(),
        }


def build_active_mask(state_raw, fields, mask_cfg):
    """
    Build per-channel boolean active mask from raw (unnormalized) state data.

    Args:
        state_raw: numpy array or torch tensor, shape [..., C] (raw physical values).
                   Typically [T, N, C] from dataset or [B, T, N, C] from batch.
        fields: list of field names, e.g. ["T", "alpha.air"]
        mask_cfg: dict with "field" and "threshold" keys, or None.
                  Example: {"field": ["T", "alpha.air"], "threshold": [800, [0.4, 0.6]]}

    Returns:
        active_mask: bool tensor same shape as state_raw, True where active.
                     None if mask_cfg is None/empty or has no valid field+threshold.
    """
    if not mask_cfg:
        return None
    mask_fields = mask_cfg.get("field", [])
    thresholds = mask_cfg.get("threshold", [])
    if not mask_fields or not thresholds:
        return None

    if isinstance(state_raw, np.ndarray):
        state_t = torch.from_numpy(state_raw)
    else:
        state_t = state_raw

    mask = torch.zeros_like(state_t, dtype=torch.bool)

    for i, fname in enumerate(fields):
        if fname not in mask_fields:
            continue
        cfg_idx = mask_fields.index(fname)
        if cfg_idx >= len(thresholds):
            continue
        thresh = thresholds[cfg_idx]
        ch = state_t[..., i:i+1]
        if isinstance(thresh, (list, tuple)):
            mask[..., i:i+1] = (ch > thresh[0]) & (ch < thresh[1])
        else:
            mask[..., i:i+1] = ch > thresh

    return mask


def load_json_config(path: str):
    """加载JSON配置并转为SimpleNamespace，便于点号访问。"""
    with open(path, "r") as f:
        config = json.load(f)
    return SimpleNamespace(**config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/aerogto_base.json", help="配置文件路径")
    return parser.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_vtk_result(save_dir, epoch, file_id, predictions, ground_truths, node_pos, field_names):
    """
    将预测结果保存为 VTK 文件序列
    Args:
        save_dir: 保存根目录
        epoch: 当前轮次
        file_id: 当前可视化的样本ID
        predictions: 模型预测值 [Horizon, N, C] (已反归一化)
        ground_truths: 真实值 [Horizon, N, C] (已反归一化)
        node_pos: 节点坐标 [N, 3]
        field_names: 物理场名称列表 (如 ['T', 'Ux', ...])
    """
    sample_dir = os.path.join(save_dir, "viz", f"epoch_{epoch}", file_id)

    os.makedirs(sample_dir, exist_ok=True)

    preds = predictions.detach().cpu().numpy()
    gts = ground_truths.detach().cpu().numpy()
    coords = node_pos.detach().cpu().numpy()   
    horizon = preds.shape[0]
    cloud = pv.PolyData(coords)

    for t in range(horizon):
        for i, field in enumerate(field_names):
            cloud.point_data[f"Pred_{field}"] = preds[t, :, i]
            cloud.point_data[f"True_{field}"] = gts[t, :, i]
            cloud.point_data[f"Err_{field}"] = np.abs(preds[t, :, i] - gts[t, :, i])
        cloud.save(os.path.join(sample_dir, f"step_{t:03d}.vtk"))

