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
        # print(self.mean, self.std, tensor.max(), tensor.min())
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

class ChannelNormalizerWithLog:
    """支持部分通道 Log 变换 + 均值方差归一化的工具。"""
    """
    log_fields = ["Ux", "Uy", "Uz"] 
    # 生成 log_flags 列表 [False, True, True, True, ...]
    log_flags = [f in log_fields for f in self.fields]
    """
    def __init__(self, mean, std, log_flags: list = None, log_scale: float=1e-3, eps: float = 1e-6):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, -1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 1, -1)
        self.log_scale = log_scale
        self.eps = eps
        if log_flags is not None:
            self.log_mask = torch.tensor(log_flags, dtype=torch.bool).view(1, 1, -1)
        else:
            self.log_mask = None

    def _log_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行 Log 变换: y = sign(x) * log(1 + |x| / scale)"""
        if self.log_mask is None:
            return tensor
        
        out = tensor.clone()

        mask = self.log_mask.to(tensor.device)

        target_data = out[..., mask.squeeze()] 

        transformed = torch.sign(target_data) * torch.log1p(torch.abs(target_data) / self.log_scale)

        out[..., mask.squeeze()] = transformed

        return out

    def _log_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行 Log逆变换: x = sign(y) * (exp(|y|) - 1)"""
        if self.log_mask is None:
            return tensor
            
        out = tensor.clone()
        mask = self.log_mask.to(tensor.device)
        target_data = out[..., mask.squeeze()]

        restored = torch.sign(target_data) * torch.expm1(torch.abs(target_data)) * self.log_scale
        
        out[..., mask.squeeze()] = restored
        return out

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean.to(tensor.device), self.std.to(tensor.device)
        x = self._log_forward(tensor)
        return (x - mean) / (std + self.eps)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean.to(tensor.device), self.std.to(tensor.device)
        x = tensor * (std + self.eps) + mean
        return self._log_inverse(x)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if self.log_mask is not None:
            self.log_mask = self.log_mask.to(device)
        return self
    
    def as_dict(self):
        return {
            "mean": self.mean.cpu().numpy().reshape(-1).tolist(),
            "std": self.std.cpu().numpy().reshape(-1).tolist(),
        }

class ChannelNormalizerWithArcSinh:
    """支持部分通道 ArcSinh (反双曲正弦) 变换 + 均值方差归一化的工具。"""
    
    def __init__(self, mean, std, transform_flags: list = None, scale: float = 1.0, eps: float = 1e-6):
        """
        Args:
            mean: 经过 ArcSinh 变换后统计出的均值
            std:  经过 ArcSinh 变换后统计出的标准差
            transform_flags: 布尔列表，指示哪些通道需要进行变换 (如 [False, True, True, True, False])
            scale: 控制线性到对数转换阈值的缩放因子。
                   在 scale 附近，变换由线性平滑过渡为对数。
        """
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, -1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 1, -1)
        self.scale = scale
        self.eps = eps
        
        if transform_flags is not None:
            self.transform_mask = torch.tensor(transform_flags, dtype=torch.bool).view(1, 1, -1)
        else:
            self.transform_mask = None

    def _arcsinh_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行 ArcSinh 变换: y = arcsinh(x / scale)"""
        if self.transform_mask is None:
            return tensor
        
        out = tensor.clone()
        mask = self.transform_mask.to(tensor.device)
        
        target_data = out[..., mask.squeeze()] 
        # ArcSinh 天然支持正负号，无需手动 sign 提取
        transformed = torch.arcsinh(target_data / self.scale)
        out[..., mask.squeeze()] = transformed
        
        return out

    def _arcsinh_inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行 ArcSinh 逆变换: x = scale * sinh(y)"""
        if self.transform_mask is None:
            return tensor
            
        out = tensor.clone()
        mask = self.transform_mask.to(tensor.device)
        
        target_data = out[..., mask.squeeze()]
        restored = self.scale * torch.sinh(target_data)
        out[..., mask.squeeze()] = restored
        
        return out

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        # 第一步：先对长尾物理场进行整容（压缩极端值）
        x = self._arcsinh_forward(tensor)
        # 第二步：对整容后的数据进行标准 Z-score，对齐神经网络权重区间
        mean, std = self.mean.to(tensor.device), self.std.to(tensor.device)
        return (x - mean) / (std + self.eps)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        # 第一步：先还原 Z-score 标准化
        mean, std = self.mean.to(tensor.device), self.std.to(tensor.device)
        x = tensor * (std + self.eps) + mean
        # 第二步：将数据从压缩空间还原回真实的物理空间
        return self._arcsinh_inverse(x)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if self.transform_mask is not None:
            self.transform_mask = self.transform_mask.to(device)
        return self
    
    def as_dict(self):
        return {
            "mean": self.mean.cpu().numpy().reshape(-1).tolist(),
            "std": self.std.cpu().numpy().reshape(-1).tolist(),
            "scale": self.scale
        }
    

def load_json_config(path: str):
    """加载JSON配置并转为SimpleNamespace，便于点号访问。"""
    with open(path, "r") as f:
        config = json.load(f)
    return SimpleNamespace(**config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/aerogto_base.json", help="配置文件路径")
    return parser.parse_args()

def parse_args_default(default=None):
    parser = argparse.ArgumentParser()
    if default is None:
        default = "config/aerogto_base.json"
    parser.add_argument("--config", type=str, default=default, help="配置文件路径")
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

