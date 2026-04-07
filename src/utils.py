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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def collate_variable_nodes(batch: list) -> dict:
    """DataLoader collate_fn for samples with varying node counts across files.

    Tensors indexed by nodes (N dim) are padded to the max N in the batch.
    Edges are offset per sample so node indices remain correct after stacking.

    Padded positions:
      - state, node_pos, node_type, active_mask → padded with 0
      - edges                                   → offset added per sample, no padding needed (concat)

    Returns a dict where:
      - state:       [B, T, N_max, C]
      - node_pos:    [B, N_max, 3]
      - node_type:   [B, N_max, 1]
      - edges:       [B, E_max, 2]  (padded with -1 rows)
      - node_mask:   [B, N_max]     bool, True for real nodes
      - conditions:  [B, cond_dim]
      - time_seq:    [B, horizon, 1]
      - grid_shape:  [B, 3]
      - spatial_inform: [B, 9]  (if present)
      - active_mask: [B, T, N_max, C]  (if present)
      - dt, scalar values stacked as [B]
    """
    # --- collect sizes ---
    node_counts = [s["node_pos"].shape[0] for s in batch]
    edge_counts = [s["edges"].shape[0] for s in batch]
    N_max = max(node_counts)
    E_max = max(edge_counts)
    B = len(batch)

    # --- keys that need node-dim padding (dim that equals N) ---
    # state: [T, N, C]  → pad dim 1
    # node_pos: [N, 3]  → pad dim 0
    # node_type: [N, 1] → pad dim 0
    # active_mask: [T, N, C] → pad dim 1

    def pad_node_dim(tensor, N_max, node_dim):
        """Zero-pad `tensor` along `node_dim` to length N_max."""
        pad_size = N_max - tensor.shape[node_dim]
        if pad_size == 0:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[node_dim] = pad_size
        return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=node_dim)

    # --- node_mask ---
    node_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, n in enumerate(node_counts):
        node_mask[i, :n] = True

    # --- state ---
    state_list = [pad_node_dim(s["state"], N_max, node_dim=1) for s in batch]
    state = torch.stack(state_list, dim=0)  # [B, T, N_max, C]

    # --- node_pos ---
    node_pos_list = [pad_node_dim(s["node_pos"], N_max, node_dim=0) for s in batch]
    node_pos = torch.stack(node_pos_list, dim=0)  # [B, N_max, 3]

    # --- node_type ---
    node_type_list = [pad_node_dim(s["node_type"], N_max, node_dim=0) for s in batch]
    node_type = torch.stack(node_type_list, dim=0)  # [B, N_max, 1]

    # --- edges: keep per-sample local indices, pad short rows with -1 ---
    edge_list = []
    for i, s in enumerate(batch):
        e = s["edges"].long()
        pad_rows = E_max - e.shape[0]
        if pad_rows > 0:
            e = torch.cat([e, torch.full((pad_rows, 2), -1, dtype=torch.long)], dim=0)
        edge_list.append(e)
    edges = torch.stack(edge_list, dim=0)  # [B, E_max, 2]

    # --- scalar / fixed-size tensors ---
    conditions = torch.stack([s["conditions"] for s in batch], dim=0)
    time_seq   = torch.stack([s["time_seq"]   for s in batch], dim=0)
    grid_shape = torch.stack([s["grid_shape"] for s in batch], dim=0)
    dt         = torch.tensor([s["dt"]        for s in batch], dtype=torch.float32)

    out = {
        "state":      state,
        "node_pos":   node_pos,
        "node_type":  node_type,
        "node_mask":  node_mask,
        "edges":      edges,
        "conditions": conditions,
        "time_seq":   time_seq,
        "grid_shape": grid_shape,
        "dt":         dt,
    }

    if "spatial_inform" in batch[0]:
        out["spatial_inform"] = torch.stack([s["spatial_inform"] for s in batch], dim=0)

    if "active_mask" in batch[0]:
        am_list = [pad_node_dim(s["active_mask"], N_max, node_dim=1) for s in batch]
        out["active_mask"] = torch.stack(am_list, dim=0)

    return out


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

