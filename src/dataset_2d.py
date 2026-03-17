import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import ChannelNormalizer


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


def _find_closest_z_layer(f: h5py.File, grid_shape: Tuple[int, int, int], target_z: float):
    """
    在三维网格中找到 Z 坐标最接近 target_z 的层索引。
    假设网格是结构化的，且 Z 轴是变化最慢的维度 (index = x + nx*y + nx*ny*z)。
    """
    gx, gy, gz = grid_shape
    stride_z = gx * gy
    
    best_k = -1
    min_dist = 1e-8
    
    # 遍历每一层的第一个点，检查其 Z 坐标
    # 注意：为了效率，我们不读取整个 point 数据集，而是跳跃读取
    # 如果文件很大，这种循环读取可能稍慢，但对于一般 LPBF 数据集 (gz < 200) 是瞬间完成的
    for k in range(gz):
        idx = k * stride_z
        # 读取该层第一个点的坐标 [x, y, z]
        pt = f["point"][idx] 
        z_val = pt[2]
        
        dist = abs(z_val - target_z)
        if dist < min_dist:
            min_dist = dist
            best_k = k
            
    # print(f"   -> [2D Slice] Target Z={target_z:.2e}, Found Z layer {best_k} (approx {f['point'][best_k*stride_z][2]:.2e})")
    return best_k


def _compute_2d_indices(grid_shape: Tuple[int, int, int], stride: Tuple[int, int, int], z_layer_idx: int):
    """
    生成特定 Z 层的二维索引。
    """
    gx, gy, gz = grid_shape
    sx, sy, sz = stride # sz 在这里仅用于兼容接口，实际上 Z 方向不采样

    # 仅在 X 和 Y 方向生成索引
    xs = list(range(0, gx, sx))
    ys = list(range(0, gy, sy))
    
    # 保证包含边界
    if xs[-1] != gx - 1: xs.append(gx - 1)
    if ys[-1] != gy - 1: ys.append(gy - 1)
    
    # 二维形状 (Nx, Ny)
    ds_shape = (len(xs), len(ys))
    
    indices = []
    # 固定 Z 层基准索引
    base_z = z_layer_idx * gx * gy
    
    for y in ys:
        base_y = base_z + y * gx
        for x in xs:
            indices.append(base_y + x)

    return np.asarray(indices, dtype=np.int32), ds_shape


def _build_2d_edges(ds_shape: Tuple[int, int], sample_ratio=1.0) -> torch.Tensor:
    """
    基于二维规则网格生成四向邻接边 (上下左右)。
    ds_shape: (nx, ny)
    """
    nx, ny = ds_shape
    edges = []

    def idx(x, y):
        return x + nx * y # 二维索引展平

    for y in range(ny):
        for x in range(nx):
            cur = idx(x, y)
            # 右
            if x + 1 < nx:
                edges.append((cur, idx(x + 1, y)))
            # 上
            if y + 1 < ny:
                edges.append((cur, idx(x, y + 1)))

    edges_arr = np.asarray(edges, dtype=np.int32)
    # 无向图去重 (sort + unique)
    edges_arr = np.sort(edges_arr, axis=1)
    edges_arr = np.unique(edges_arr, axis=0)

    if sample_ratio < 1.0:
        total_edges = edges_arr.shape[0]
        target_num = int(total_edges * sample_ratio)
        sample_indices = np.random.choice(total_edges, target_num, replace=False)
        edges_arr = edges_arr[sample_indices]

    return torch.from_numpy(edges_arr)


def _build_2d_node_type(ds_shape: Tuple[int, int]) -> torch.Tensor:
    """二维边界标记：矩形边框为 1，内部为 0。"""
    nx, ny = ds_shape
    node_types = np.zeros((nx * ny, 1), dtype=np.int32)
    for y in range(ny):
        for x in range(nx):
            # 这里的边界是二维平面的边缘
            if x in (0, nx - 1) or y in (0, ny - 1):
                node_types[x + nx * y, 0] = 1
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
    """(保留原始逻辑) 将全局参数、场信息、边界条件压平成一个条件向量。"""
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
    """按通道统计均值和方差。"""
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
            data = np.stack(phys_data, axis=-1).reshape(-1, num_channels)

            mean += data.sum(axis=0)
            sq += (data**2).sum(axis=0)
            count += data.shape[0]

        mean = mean / count
        std = np.sqrt(np.maximum(sq / count - mean**2, 1e-12))
        std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)

def scale_pos(node_pos):
        
        xx = node_pos[...,0]
        yy = node_pos[...,1]

        x_norm = (xx - xx.min()) / (xx.max() - xx.min())
        y_norm = (yy - yy.min()) / (yy.max() - yy.min())
        
        node_pos_new = torch.stack((x_norm, y_norm), dim=-1)
        return node_pos_new

class AeroGtoDataset2D(Dataset):
    """
    针对 LPBF 数据的 2D 切片数据集。
    提取特定 Z 高度 (slice_z) 的平面数据进行训练。
    """

    def __init__(
        self,
        file_list: Iterable[str],
        mode: str = "train",
        fields: List['str'] = ['T'],
        input_steps: int = 1,
        horizon: int = 5,
        time_stride: int = 1,
        spatial_stride: Union[int, Tuple[int, int, int]] = 1,
        normalize: bool = True,
        samples_per_file: int = 32,
        norm_cache: Optional[str] = None,
        slice_z: float = 5e-4,  # [新增] 目标切片高度
        mat_data = None,
    ):
        super().__init__()
        assert mode in {"train", "test"}, "mode 只能为 train 或 test"
        self.mode = mode
        self.fields = fields
        self.input_steps = input_steps
        self.horizon = horizon
        self.time_stride = time_stride
        self.spatial_stride = _normalize_stride(spatial_stride)
        self.normalize = normalize
        self.mat_mean_and_std = mat_data
        self.samples_per_file = samples_per_file
        self.norm_cache = norm_cache
        self.slice_z = slice_z  # 记录目标 Z

        self.file_paths = _read_file_list(file_list)
        self.meta_cache = {}
        self.sample_keys = []
        self.max_start_per_file = []
        
        for file_id, path in enumerate(self.file_paths):
            meta = self._build_meta(path)
            self.meta_cache[path] = meta
            self.max_start_per_file.append(meta["max_start"])

            if mode == "train":
                for _ in range(samples_per_file):
                    self.sample_keys.append((file_id, None))
            else:
                step = max(1, horizon // 2)
                for start in range(1, meta["max_start"] + 1, step):
                    self.sample_keys.append((file_id, start))

        example_meta = next(iter(self.meta_cache.values()))
        self.cond_dim = example_meta["conditions"].shape[-1]
        self.node_num = example_meta["node_pos"].shape[0]
        self.dt = example_meta["dt"]
        
        # 2D 数据通常比 3D 小很多，注意 BatchSize 可以适当调大
        num_channels = len(self.fields)
        if self.normalize and self.mode == "train":
            self.normalizer = self._load_or_compute_normalizer()
        else:
            self.normalizer = ChannelNormalizer(np.zeros(num_channels, dtype=np.float32), np.ones(num_channels, dtype=np.float32))

    def _load_or_compute_normalizer(self) -> ChannelNormalizer:
        # 缓存 key 增加 slice_z 信息，防止混淆 2D 和 3D 的统计量
        cache_path = Path(self.norm_cache) if self.norm_cache else None
        field_hash = "-".join(self.fields)
        cache_key = f"{self.file_paths[0]}|stride={self.spatial_stride}|fields={field_hash}|z={self.slice_z}"
        
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
                if cache_key in cache:
                    stats = cache[cache_key]
                    if len(stats["mean"]) == len(self.fields):
                        print(f"   -> Loaded normalizer from cache.")
                        return ChannelNormalizer(stats["mean"], stats["std"])
            except Exception:
                pass 

        print(f"   -> Computing stats for 2D slice...")
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

    def _build_meta(self, path: str):
        path = str(Path(path).expanduser().resolve())
        with h5py.File(path, "r") as f:

            block = f["mesh/block"][0].astype(int)
            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)
            z_layer_idx = _find_closest_z_layer(f, grid_shape, self.slice_z)
            indices, ds_shape = _compute_2d_indices(grid_shape, self.spatial_stride, z_layer_idx)

            # 4. 加载点坐标 (N, 3) - 注意这里依然是3维坐标，但Z值是常数
            point_all = f["point"][:]
            point = point_all[indices]
            point_2d = point[:, :2]
            node_pos = torch.from_numpy(point_2d.astype(np.float32))

            # 5. 构建 2D 边
            edges = _build_2d_edges(ds_shape)
            node_type = _build_2d_node_type(ds_shape)

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
            "conditions": conditions,
            "dt": dt,
            "max_start": max_start,
        }

    def __len__(self):
        return len(self.sample_keys)

    def _load_window(self, path: str, indices: np.ndarray, start: int):
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
        state = torch.from_numpy(state_np) 
        if self.normalize:
            state = self.normalizer.normalize(state)
            node_pos = scale_pos(meta["node_pos"])

        rel_time = time_seq[1:] - time_seq[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)
        sample = {
            "dt": meta['dt'],
            "state": state,  
            "time_seq": time_tensor,  
            "node_pos": node_pos,
            "edges": meta["edges"],
            "node_type": meta["node_type"],
            "conditions": meta["conditions"],
        }
        return sample