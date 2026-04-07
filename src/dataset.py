import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import ChannelNormalizer, build_active_mask


def _read_file_list(file_list: Iterable[str]) -> List[str]:
    """读取txt列表或直接的路径列表，返回绝对路径列表。"""
    paths = []
    for item in file_list:
        p = Path(item)
        if p.is_file() and p.suffix in {".txt", ".list"}:
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        paths.append(str(Path(line).expanduser().resolve()))
        else:
            paths.append(str(p.expanduser().resolve()))
    if not paths:
        raise ValueError("未找到有效的数据文件路径")
    return paths


def _normalize_stride(stride) -> Tuple[int, int, int]:
    if isinstance(stride, int):
        return (stride, stride, stride)
    if isinstance(stride, (list, tuple)) and len(stride) == 3:
        return tuple(int(x) for x in stride)
    raise ValueError("spatial_stride 需要是int或长度为3的list/tuple")


def _compute_downsample_indices(grid_shape: Tuple[int, int, int], stride: Tuple[int, int, int]):
    """根据网格尺寸和步长生成下采样索引以及下采样后的shape。"""
    gx, gy, gz = grid_shape
    sx, sy, sz = stride

    xs = list(range(0, gx, sx))
    ys = list(range(0, gy, sy))
    zs = list(range(0, gz, sz))
    if xs[-1] != gx - 1:
        xs.append(gx - 1)
    if ys[-1] != gy - 1:
        ys.append(gy - 1)
    if zs[-1] != gz - 1:
        zs.append(gz - 1)

    ds_shape = (len(xs), len(ys), len(zs))
    indices = []
    for z in zs:
        for y in ys:
            base = z * gx * gy + y * gx
            for x in xs:
                indices.append(base + x)

    return np.asarray(indices, dtype=np.int32), ds_shape


def _build_grid_edges(ds_shape: Tuple[int, int, int], sample_ratio=1.0) -> torch.Tensor:
    """基于规则网格生成六向邻接边。"""
    nx, ny, nz = ds_shape
    edges = []
    def idx(x, y, z):
        return x + nx * y + nx * ny * z

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                cur = idx(x, y, z)
                if x + 1 < nx:
                    edges.append((cur, idx(x + 1, y, z)))
                if y + 1 < ny:
                    edges.append((cur, idx(x, y + 1, z)))
                if z + 1 < nz:
                    edges.append((cur, idx(x, y, z + 1)))

    edges_arr = np.asarray(edges, dtype=np.int32)
    # 保证无向去重
    edges_arr = np.sort(edges_arr, axis=1)
    edges_arr = np.unique(edges_arr, axis=0)

    if sample_ratio < 1.0:
        total_edges = edges_arr.shape[0]
        target_num = int(total_edges * sample_ratio)
        # print(f"[Warning] 正在对边进行下采样: {total_edges} -> {target_num} (Ratio={sample_ratio})")   
        sample_indices = np.random.choice(total_edges, target_num, replace=False)
        edges_arr = edges_arr[sample_indices]

    return torch.from_numpy(edges_arr)


def _build_node_type(ds_shape: Tuple[int, int, int], y_divide=17) -> torch.Tensor:
    """0: 内部节点，1: 固相边界节点，2:液相边界节点"""
    nx, ny, nz = ds_shape
    node_types = np.zeros((nx * ny * nz, 1), dtype=np.int32)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                idx = x + nx * y + nx * ny * z
                if y == 0: node_types[idx] = 1
                elif y == ny - 1: node_types[idx] = 2
                else:
                    if x in (0, nx - 1) or z in (0, nz - 1):
                        if y <= y_divide: node_types[idx] = 1
                        else: node_types[idx] = 2
    return torch.from_numpy(node_types)

def _process_condition_normalize(f: h5py.File, mat_mean_and_std=None) -> np.ndarray:
    """对参数进行标准化操作，数据集中不变的参数暂时不传入模型"""
    cond_list = []
    thermal_cond_list = [
    ("parameter/thermal", 3, np.arange(100, 300, 10).mean(), np.arange(100, 300, 10).std()),    # 激光功率
    ("parameter/thermal", 4, np.arange(35e-6, 45e-6, 1e-6).mean(), np.arange(35e-6, 45e-6, 1e-6).std()),   # 激光半径
    ("parameter/thermal", 5, np.arange(2e-4, 7e-4, 1e-4).mean(), np.arange(2e-4, 7e-4, 1e-4).std()), # 激光起始 x 坐标
    ("parameter/thermal", 7, np.arange(0.35, 0.45, 0.01).mean(), np.arange(0.35, 0.45, 0.01).std()), # 吸收率
    ("parameter/thermal", 8, np.arange(0.2, 0.4, 0.01).mean(), np.arange(0.2, 0.4, 0.01).std()), # 能量移动速度】
    ]

    for path, idx, mean_val, std_val in thermal_cond_list:
        if path in f:
            data_arr = f[path][:]
            val = data_arr.reshape(-1)[idx]
            val_norm = (float(val) - mean_val) / (std_val + 1e-6)
            cond_list.append([val_norm])
        else:
            cond_list.append([0.0])
    # 针对 material 的处理 （3*15）的数组
    mat_path = "parameter/material"
    mat_all = f[mat_path][:]
    mat_2 = mat_all[0:-1, :]
    if mat_mean_and_std is None:
        mat_mean = np.mean(mat_2, axis=0)
        mat_std = np.std(mat_2, axis=0)
    else:
        mat_mean, mat_std = mat_mean_and_std

    mat_norm = (mat_2 - mat_mean) / (mat_std + 1e-6)
    cond_list.append(mat_norm.reshape(-1))

    dump = len(np.unique(f["parameter/dump"][:][:, -1]))
    cond_list.append([dump])

    if not cond_list:
        return np.array([], dtype=np.float32)

    return np.concatenate(cond_list, axis=0).astype(np.float32), (mat_mean, mat_std)

def _condition_vector(f: h5py.File, field_names: List[str]) -> np.ndarray:
    """将全局参数、场信息、边界条件压平成一个条件向量。"""
    thermal = f["parameter/thermal"][:].reshape(-1)
    material = f["parameter/material"][:].reshape(-1)
    interact = f["parameter/interact"][:].reshape(-1)

    dump = f["parameter/dump"][:]
    dump_mean = dump.mean(axis=0)
    dump_std = dump.std(axis=0)

    field_box = f["field/box"][:].reshape(-1)
    field_scalar = f["field/scalar"][:].reshape(-1)
    field_velocity = f["field/velocity"][:].reshape(-1)

    inicond_list, boundcond_list = [], []
    for fname in field_names:
        if fname in f["inicond"]:
            data = f["inicond"][fname][:].reshape(-1)
        inicond_list.append(data)
        if fname in f['boundcond']:
            data = f["boundcond"][fname][:].reshape(-1)
        boundcond_list.append(data)

    inicond = np.concatenate(inicond_list, axis=0)
    boundcond = np.concatenate(boundcond_list, axis=0)

    cond_vec = np.concatenate(
        [
            thermal,
            material,
            interact,
            dump_mean,
            dump_std,
            field_box,
            field_scalar,
            field_velocity,
            inicond,
            boundcond,
        ],
        axis=0,
    ).astype(np.float32)

    return cond_vec


def _compute_stats(path: str, indices: np.ndarray, field_names: List[str], chunk: int = 10):
    """按通道统计均值和方差，避免一次性载入全部数据。"""
    num_channels = len(field_names)

    with h5py.File(path, "r") as f:
        total_t = f[f'state/{field_names[0]}'].shape[0]
        mean, sq = np.zeros(num_channels, dtype=np.float64), np.zeros(num_channels, dtype=np.float64)
        count = 0        

        for start in range(0, total_t, chunk):
            end = min(total_t, start + chunk)
            phys_data = []
            for fname in field_names:
                fkey = f"state/{fname}"
                full_chunk = f[fkey][start:end]
                d = full_chunk[:, indices, 0]
                phys_data.append(d)
            data = np.stack(phys_data, axis=-1).reshape(-1, num_channels) # [Points, Channels]

            mean += data.sum(axis=0)
            sq += (data**2).sum(axis=0)
            count += data.shape[0]

        mean = mean / count
        std = np.sqrt(np.maximum(sq / count - mean**2, 1e-12))
        std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


class AeroGtoDataset(Dataset):
    """面向LPBF数据的自回归训练集封装。"""

    def __init__(
        self, args,
        mode: str = "train",
        mat_data = None,
    ):
        super().__init__()
        assert mode in {"train", "test"}, "mode 只能为 train 或 test"
        data_cfg = args.data
        self.config = data_cfg
        self.mode = mode

        self.fields = data_cfg.get("fields", ["T"])
        self.input_steps = data_cfg.get("input_steps", 1)
        self.horizon = data_cfg.get(f"horizon_{mode}", 1)
        self.time_stride = data_cfg.get("time_stride", 1)
        self.spatial_stride = _normalize_stride(data_cfg.get("spatial_stride", 1))
        self.normalize = data_cfg.get("normalize", True)
        self.mat_mean_and_std = mat_data
        self.samples_per_file =  data_cfg.get("samples_per_file", 32)
        self.norm_cache = data_cfg.get("norm_cache")

        self.mask_cfg = args.train.get("weight_loss", {"field": ["T", "alpha.air"], "threshold": [800, [0.4, 0.6]]}) # 临时

        self.file_paths = _read_file_list(data_cfg[f"{mode}_list"])
        self.meta_cache = {}
        self.sample_keys = []
        self.max_start_per_file = []
        self.time_ref = 2e-5 if self.config.get("dt_scale", False) else 1
        self.edge_sample_ratio = self.config.get("edge_sample_ratio", 1.0)

        for file_id, path in enumerate(self.file_paths):
            meta = self._build_meta(path)
            self.meta_cache[path] = meta
            self.max_start_per_file.append(meta["max_start"])

            if mode == "train":
                for _ in range(self.samples_per_file):
                    self.sample_keys.append((file_id, None))  # None 表示随机起点
            else:
                # 测试阶段均匀取样，覆盖全序列
                step = max(1, self.horizon // 2)
                for start in range(1, meta["max_start"] + 1, step):
                    self.sample_keys.append((file_id, start))

        example_meta = next(iter(self.meta_cache.values()))
        self.cond_dim = example_meta["conditions"].shape[-1]
        self.node_num = example_meta["node_pos"].shape[0]
        self.dt = example_meta["dt"] / self.time_ref
        num_channels = len(self.fields)
        
        if self.normalize and self.mode == "train":
            # self.normalizer = self._load_or_compute_normalizer()
            self.normalizer = self._load_normalizer()
        else:
            self.normalizer = ChannelNormalizer(np.zeros(num_channels, dtype=np.float32), np.ones(num_channels, dtype=np.float32))

    def _load_or_compute_normalizer(self) -> ChannelNormalizer:
        cache_path = Path(self.norm_cache) if self.norm_cache else None
        field_hash = "-".join(self.fields)
        cache_key = f"{self.file_paths[0]}|stride={self.spatial_stride}|fields={field_hash}"
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
                if cache_key in cache:
                    stats = cache[cache_key]
                    if len(stats["mean"]) == len(self.fields):
                        return ChannelNormalizer(stats["mean"], stats["std"])
            except Exception:
                pass  # 缓存损坏则重新计算

        mean, std = _compute_stats(self.file_paths[0], self.meta_cache[self.file_paths[0]]["indices"], self.fields)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache = {}
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        cache = json.load(f)
                except Exception:
                    cache = {}
            cache[cache_key] = {"mean": mean.tolist(), "std": std.tolist()}
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)

        return ChannelNormalizer(mean, std)

    def _load_normalizer(self) -> ChannelNormalizer:
        
        field_stats_config = {
            "T":         (5.2999e+02, 4.5454e+02),  
            "Ux":        (4.0041e-05, 2.4173e-01),
            "Uy":        (-1.6900e-05, 2.5172e-01),
            "Uz":        (3.3602e-07, 1.1976e-01),
            "alpha.air": (0, 1), # (3.6361e-01, 4.6604e-01)    
            "alpha.titanium": (0, 1), # (6.3635e-01, 4.6607e-01)
            "gamma_liquid": (0, 1)} # (2.9447e-02, 1.6011e-01)
        
        mean_list, std_list = [], []
        for fname in self.fields:
            m, s = field_stats_config[fname]
            mean_list.append(m)
            std_list.append(s)
        mean_arr = np.array(mean_list, dtype=np.float32)
        std_arr = np.array(std_list, dtype=np.float32)

        return ChannelNormalizer(mean_arr, std_arr)


    def scale_3D_pos(self, node_pos):
        
        xx = node_pos[...,0]
        yy = node_pos[...,1]
        zz = node_pos[...,2]

        x_norm = (xx - xx.min()) / (xx.max() - xx.min())
        y_norm = (yy - yy.min()) / (yy.max() - yy.min())
        z_norm = (zz - zz.min()) / (zz.max() - zz.min())
        
        node_pos_new = torch.stack((x_norm, y_norm, z_norm), dim=-1)
        return node_pos_new

    def _build_meta(self, path: str):
        path = str(Path(path).expanduser().resolve())
        with h5py.File(path, "r") as f:
            # 网格尺寸与下采样索引
            block = f["mesh/block"][0].astype(int)
            bound = f["mesh/bound"][:].astype(np.float32)  # [3, 2] 不同维度的坐标范围
            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)  # 点的数量
            indices, ds_shape = _compute_downsample_indices(grid_shape, self.spatial_stride)

            spatial_inform = torch.from_numpy(
                np.concatenate([bound.flatten(), np.array(ds_shape, dtype=np.float32), np.array([self.time_ref], dtype=np.float32)])
            )  # shape [10]: 6项坐标范围 + 3项下采样后网格数量 + 1项时间参考值

            point_all = f["point"][:]
            point = point_all[indices]

            node_pos = torch.from_numpy(point.astype(np.float32))

            edges = _build_grid_edges(ds_shape, self.edge_sample_ratio)
            node_type = _build_node_type(ds_shape)

            if self.normalize:
                if self.mat_mean_and_std is None:
                    conditions, self.mat_mean_and_std = _process_condition_normalize(f)
                else:
                    conditions, _ = _process_condition_normalize(f, self.mat_mean_and_std)
            else:
                conditions = _condition_vector(f, self.fields)
            
            conditions = torch.from_numpy(conditions)

            time_all = f["time"][:]
            dt = np.float32(np.mean(np.diff(time_all)))
            total_steps = len(time_all)
            max_start = total_steps - (self.input_steps + self.horizon * self.time_stride)
            if max_start < 0:
                raise ValueError(f"时间窗口超出范围，total_steps={total_steps}")

        return {
            "grid_shape": grid_shape,
            "indices": indices,
            "ds_shape": ds_shape,
            "node_pos": node_pos,
            "edges": edges,
            "node_type": node_type,
            "spatial_inform": spatial_inform,
            "conditions": conditions,
            "dt": dt,
            "max_start": max_start,
        }

    def __len__(self):
        return len(self.sample_keys)

    def _load_window(self, path: str, indices: np.ndarray, start: int):
        """读取 [start, start + horizon] 对应的状态与时间。"""
        with h5py.File(path, "r") as f:
            time_idx = start + np.arange(0, self.horizon + 1) * self.time_stride
            channels = []
            for fname in self.fields:
                fkey = f"state/{fname}"
                data_all_points = f[fkey][time_idx]
                d = data_all_points[:, indices, 0]
                channels.append(d)
            state = np.stack(channels, axis=-1).astype(np.float32)
            time_all = f["time"][time_idx]
        return state, time_all

    def __getitem__(self, idx):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])

        state_np, time_seq = self._load_window(path, meta["indices"], start_idx)
        state = torch.from_numpy(state_np)  # [T, N, 4]

        active_mask = build_active_mask(state_np, self.fields, self.mask_cfg)

        if self.normalize:
            state = self.normalizer.normalize(state)
            node_pos = self.scale_3D_pos(meta["node_pos"])
        else:
            node_pos = meta["node_pos"]

        # 目标时间步（相对起始时刻）
        rel_time = time_seq[1:] - time_seq[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)
        sample = {
            "dt": meta['dt'] * self.time_stride / self.time_ref,
            "state": state,  # [1 + horizon, N, 4]
            "time_seq": time_tensor / self.time_ref,  # [horizon, 1]
            "node_pos": node_pos,
            "edges": meta["edges"],
            "node_type": meta["node_type"],
            "conditions": meta["conditions"],
            "grid_shape": torch.tensor(list(meta["ds_shape"])),
            "spatial_inform": meta["spatial_inform"],
        }
        if active_mask is not None:
            sample["active_mask"] = active_mask
        return sample
