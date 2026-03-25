import json
import random
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from tqdm import tqdm

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import ChannelNormalizer, ChannelNormalizerWithLog, ChannelNormalizerWithArcSinh


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

def _process_condition_normalize(f: h5py.File, mat_mean_and_std=None):
    """对参数进行标准化操作，数据集中不变的参数暂时不传入模型"""
    cond_list = []
    thermal_cond_list = [
    ("parameter/thermal", 3, np.arange(100, 300, 10).mean(), np.arange(100, 300, 10).std()),    # 激光功率
    ("parameter/thermal", 4, np.arange(35e-6, 45e-6, 1e-6).mean(), np.arange(35e-6, 45e-6, 1e-6).std()),   # 激光半径
    ("parameter/thermal", 5, np.arange(2e-4, 7e-4, 1e-4).mean(), np.arange(2e-4, 7e-4, 1e-4).std()), # 激光起始 x 坐标
    ("parameter/thermal", 7, np.arange(0.35, 0.45, 0.01).mean(), np.arange(0.35, 0.45, 0.01).std()), # 吸收率
    ("parameter/thermal", 8, np.arange(0.2, 0.4, 0.01).mean(), np.arange(0.2, 0.4, 0.01).std()), # 能量移动速度
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
    mat_ori_data = np.array([[8e-07, 7400, 7.45e6, 3090, 0.05593, 1658, 1723, 684, 9.248, 0.01571, 18, 2.155e5, 3.27, 4.48, 0.36],
                    [8.7e-7, 8040, 6.4e6, 3188, 0.05975, 1578, 1623, 638, 4.5847, 0.0163, 30.078, 2.61e5, 2.88, 5.26, 0.36],
                    [1.33e-5, 1.67, 1000, 1000, 0.04, 1941, 1942, 520, 0.02, 0, 0.02, 0, 3.27, 4.48, 0.4],
                    [5.185e-7, 8100, 4.38e6, 2835, 0.0677, 1123, 1273, 416, 63, 0, 63, 1.6344e5, 0.35, 6.97, 0.36]], dtype=np.float32)
    
    mat_path = "parameter/material"
    mat_all = f[mat_path][:]
    mat_2 = mat_all[0:-1, :]

    # for i in range(mat_2.shape[0]):
    #     for k in range(len(mat_ori_data)):
    #         if np.allclose(np.sort(mat_2[i]), np.sort(mat_ori_data[k])):
    #             mat_2[i] = mat_ori_data[k]
    #             break

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

class AeroGtoDataset(Dataset):
    """面向LPBF数据的自回归训练集封装。"""

    def __init__(
        self, config,
        mode: str = "train",
        mat_data = None,
    ):
        super().__init__()
        assert mode in {"train", "test"}, "mode 只能为 train 或 test"
        self.config = config
        self.mode = mode
        self.fields = config["fields"]
        self.input_steps = config["input_steps"]
        self.horizon = config[f"horizon_{self.mode}"]
        self.time_stride = config.get("time_stride", 1)
        self.spatial_stride = _normalize_stride(config.get("spatial_stride", 1))
        self.normalize = config.get("normalize", True)
        self.mat_mean_and_std = mat_data
        self.samples_per_file = config.get("samples_per_file", 32)
        self.log_velocity = config.get("log_velocity", False)
        self.velocity_transformode = config.get("velocity_transformode", None)
        self.edge_sample_ratio = config.get("edge_sample_ratio", 1.0)

        self.file_paths = _read_file_list(config[f"{self.mode}_list"])
        self.cache_path = self._generate_cache_path(config["norm_cache"])

        # random.shuffle(self.file_paths)
        self.meta_cache = {}
        self.sample_keys = []
        self.max_start_per_file = []

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
        self.dt = example_meta["dt"]
        num_channels = len(self.fields)
        if self.normalize and self.mode == "train":
            # self.normalizer = self._load_or_compute_normalizer()
            self.normalizer = self._load_normalizer()
        else:
            self.normalizer = ChannelNormalizer(
                np.zeros(num_channels, dtype=np.float32),
                np.ones(num_channels, dtype=np.float32),
            )

    def _generate_cache_path(self, base_dir) -> Path:
        if base_dir is None:
            base_dir = "../result/norm_cache"
        base_path = Path(base_dir).expanduser().resolve()

        if base_path.suffix == '.json': # 传入具体 .json 路径则直接使用
            return base_path
        if self.file_paths:
            prefix = Path(self.file_paths[0]).stem
        else:
            prefix = "lpbf_dataset"
        
        fields_str = "-".join(self.fields) # 物理场信息拼接

        # 对文件路径进行排序并取哈希，防止仿真文件的增删
        path_str = "".join(sorted(self.file_paths)).encode('utf-8')
        file_hash = hashlib.md5(path_str).hexdigest()[:6]
    
        cache_filename = f"{prefix}_{fields_str}_{file_hash}_{self.velocity_transformode}_stats.json"
        return base_path / cache_filename

    def _load_normalizer(self):
        # 速度场 log flag，配合 ChannelNormalizerWithLog 使用
        log_fields = ["Ux", "Uy", "Uz"] 
        log_flags = [f in log_fields for f in self.fields]
        log_scale = 1e-3

        field_stats_config = {}
        
        # ========== 核心修改：动态加载或自动计算 Cache ==========
        if self.cache_path.is_file() and self.cache_path.exists():
            print(f"[Dataset] 命中归一化缓存配置: {self.cache_path.name}")
            with open(self.cache_path, "r") as f:
                config = json.load(f)
        else:
            print(f"[Dataset] 未找到缓存 {self.cache_path.name}，开始自动计算...")
            config = self._compute_and_cache_dataset_stats(
                cache_path=self.cache_path, # 传入刚才生成的路径
                max_sample_files=5,   
                window_size=5,
                velocity_dead_zone=log_scale,
                t_threshold=301.0)

        for field_name in ["Ux", "Uy", "Uz"]:
            if field_name in self.fields:
                # Use data-driven statistics; center at 0 (velocities are symmetric)
                _, s = config.get(field_name, (0.0, 1.0))
                config[field_name] = (0.0, max(s, 0.01))

        field_stats_config.update(config)

        mean_list, std_list = [], []
        for fname in self.fields:
            m, s = field_stats_config.get(fname, (0.0, 1.0))
            mean_list.append(m)
            std_list.append(s)
            
        mean_arr = np.array(mean_list, dtype=np.float32)
        std_arr = np.array(std_list, dtype=np.float32)
        
        if self.velocity_transformode == "log":
            return ChannelNormalizerWithLog(mean_arr, std_arr, log_flags=log_flags, log_scale=log_scale)
        elif self.velocity_transformode == "arcsinh":
            return ChannelNormalizerWithArcSinh(mean_arr, std_arr, log_flags)
        elif self.velocity_transformode is None:
            return ChannelNormalizer(mean_arr, std_arr)

        # field_stats_config = {
        #     "T":         (5.2999e+02, 4.5454e+02), 
        #     # "Ux":        (-1.1922e-02, 3.1306e-01),
        #     # "Uy":        (3.7590e-04, 3.1044e-01),
        #     # "Uz":        (-4.6862e-04, 2.1007e-01), 
        #     # "Ux":        (-4.1613e-03, 1.1843e+00), # (4.0041e-05, 2.4173e-01),
        #     # "Uy":        (-2.1014e-03, 1.2541e+00), # (-1.6900e-05, 2.5172e-01),
        #     # "Uz":        (-6.2484e-05, 5.9723e-01), # (3.3602e-07, 1.1976e-01),
        #     "Ux":        (0, 1), # (4.0041e-05, 2.4173e-01),
        #     "Uy":        (0, 1), # (-1.6900e-05, 2.5172e-01),
        #     "Uz":        (0, 1), # (3.3602e-07, 1.1976e-01),
        #     "alpha.air": (0, 1), # (3.6361e-01, 4.6604e-01)    
        #     "alpha.titanium": (0, 1), # (6.3635e-01, 4.6607e-01)
        #     "gamma_liquid": (0, 1)} # (2.9447e-02, 1.6011e-01)

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
            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)  # 点的数量
            indices, ds_shape = _compute_downsample_indices(grid_shape, self.spatial_stride)

            point_all = f["point"][:]
            point = point_all[indices]

            node_pos = torch.from_numpy(point.astype(np.float32))

            edges = _build_grid_edges(ds_shape, self.edge_sample_ratio)
            node_type = _build_node_type(ds_shape)

            mat_conditions = f["parameter/material"][:]

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
            "mat_conditions": mat_conditions,
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

    def _mask_gas_region(self, state: torch.Tensor) -> torch.Tensor:
        """
        根据 alpha.air 对非金属区域（气体）进行掩码处理。
        定义: alpha.air 在 [0, 0.5] 之间为金属区域 (保留数值);
              alpha.air > 0.5 为气体区域 (其他数值置0).
        """
        # 如果字段中不包含 alpha.air，无法计算掩码，直接返回
        if "alpha.air" not in self.fields:
            return state

        alpha_idx = self.fields.index("alpha.air")
        
        # 获取 alpha.air 的值
        alpha_val = state[..., alpha_idx]
        # 生成气体区域掩码 (alpha.air > 0.5 即为气体)
        gas_mask = alpha_val > 0.5
        # 复制 state 以避免原地修改带来的潜在问题
        masked_state = state.clone()
        # 遍历所有通道
        for i, field in enumerate(self.fields):
            # if field != "alpha.air":
            if field in ["Ux", "Uy", "Uz", "gamma_liquid"]:
                masked_state[..., i][gas_mask] = 0.0
        return masked_state
    
    def _load_data_with_aux(self, path: str, indices: np.ndarray, start: int):
        """
        同时读取训练用的物理场(state) 和 辅助判断用的场(alpha, gamma)
        """
        with h5py.File(path, "r") as f:
            time_idx = start + np.arange(0, self.horizon + 1) * self.time_stride

            channels = []
            for fname in self.fields:
                fkey = f"state/{fname}"
                if fkey in f:
                    d = f[fkey][time_idx][:, indices, 0]
                else:
                    d = np.zeros((len(time_idx), len(indices)), dtype=np.float32)
                channels.append(d)
            state_np = np.stack(channels, axis=-1).astype(np.float32)
            time_all = f["time"][time_idx]

            T_vals = f["state/T"][time_idx][:, indices, 0]

            alpha_vals = f["state/alpha.air"][time_idx][:, indices, 0]
            
            gamma_vals = f["state/gamma_liquid"][time_idx][:, indices, 0]

        return state_np, time_all, T_vals, alpha_vals, gamma_vals
    
    def __getitem__(self, idx):
        data_mask, zero_mask = self.config.get("mask", False), self.config.get("zero_mask", False)
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])

        # state_np, time_seq = self._load_window(path, meta["indices"], start_idx)
        
        state_np, time_seq, T_np, alpha_np, gamma_np = self._load_data_with_aux(path, meta["indices"], start_idx)

        state, T_t = torch.from_numpy(state_np), torch.from_numpy(T_np)
        alpha_t, gamma_t = torch.from_numpy(alpha_np), torch.from_numpy(gamma_np)
        node_y = meta["node_pos"][:, 1] 

        y_cutoff_mask = node_y > 1e-4; gas_mask = alpha_t > 0.6; solid_mask = (alpha_t < 1e-4) | (gamma_t < 1e-4)

        # 不再在 dataset 中裁剪物理场，改为训练过程中不计算相应区域的 loss
        velocity_dead_zone = 1e-3
        vel_indices = [i for i, field in enumerate(self.fields) if field in ["Ux", "Uy", "Uz"]]
        if len(vel_indices) > 0:
            total_vel_abs = sum([state[..., i].abs() for i in vel_indices])

            dead_zone_mask = (total_vel_abs < velocity_dead_zone).view_as(gas_mask)
            gas_mask = gas_mask | dead_zone_mask
            
        # for i, field in enumerate(self.fields):
        #     if field in ["Ux", "Uy", "Uz"]:
        #         vel = state[..., i]
        #         state[..., i] = torch.where(vel.abs() < velocity_dead_zone, torch.zeros_like(vel), vel)

        # roi_mask = (gamma_t > 1e-4) & (alpha_t < 0.99)

        gamma_t[:, y_cutoff_mask] = 0.0

        # C = len(self.fields)
        # 初始化一个[T, N, C]的权重矩阵
        # loss_weight = torch.ones((state.shape[0], state.shape[1], C), dtype=torch.float32)
        # loss_weight = torch.ones_like(state[..., 0:1])
        # loss_weight = torch.full_like(state[..., 0:1], 1e-6)

        # # 1） 熔池核心区
        # temp_mask = T_t > 301
        # # 2） 熔池核心区
        # melt_pool_mask = (gamma_t > 0.49) & (alpha_t < 0.51)
        # # 3） 整体金属区 用于预测分界线
        # metal_mask = (alpha_t > 0.49) & (alpha_t < 0.55)

        # loss_weight[roi_mask] = 0.5 # high_weight
        
        # for i, field in enumerate(self.fields):
        #     if field == "T":
        #         loss_weight[..., i] = 1e-4
        #         loss_weight[..., i][temp_mask] = 1.0
        #     elif field in ["Ux", "Uy", "Uz"]:
        #         loss_weight[..., i] = 1e-4  # 背景极低权重
        #         loss_weight[..., i][melt_pool_mask] = 1.0
        #     elif field in ["alpha.air"]: # , "alpha.titanium", "gamma_liquid"
        #         loss_weight[..., i] = 1e-4
        #         loss_weight[..., i][metal_mask] = 1.0

        if self.normalize:
            state = self.normalizer.normalize(state)
            node_pos = self.scale_3D_pos(meta["node_pos"])
        else:
            node_pos = meta["node_pos"]

        # 目标时间步（相对起始时刻）
        rel_time = time_seq[1:] - time_seq[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)
        sample = {
            "dt": meta['dt'],
            "state": state,  # [1 + horizon, N, 4]
            "time_seq": time_tensor,  # [horizon, 1]
            "node_pos": node_pos,
            "node_pos_phys": meta["node_pos"],
            "edges": meta["edges"],
            "node_type": meta["node_type"],
            "conditions": meta["conditions"],
            # "loss_weight": loss_weight, # [T, N, 1] 用于 Loss 加权
            "gas_mask": gas_mask,
            "y_cutoff_mask": y_cutoff_mask,
            "solid_mask": solid_mask
        }
        return sample

    def _compute_and_cache_dataset_stats(
        self, 
        cache_path: Path, 
        max_sample_files=5, 
        window_size=5, 
        velocity_dead_zone=1e-3, 
        t_threshold=301.0
    ):
        """内置的统计计算逻辑：如果未命中缓存，自动采样计算并保存"""
        print(f"\n[Dataset] 未找到归一化缓存，准备自动从数据集中采样计算...")
        
        # 1. 采样文件 (只从训练集中随机抽几个文件算，没必要全算)
        paths = self.file_paths
        sampled_paths = random.sample(paths, min(len(paths), max_sample_files))
        print(f"[Dataset] 数据集总文件数: {len(paths)}, 采样计算数: {len(sampled_paths)}")

        # 2. 初始化累加器 (仅针对当前设定的 self.fields)
        stats = {field: {"sum": 0.0, "sq_sum": 0.0, "count": 0} for field in self.fields}

        # 3. 遍历采样文件进行计算
        for path_str in sampled_paths:
            path = Path(path_str).expanduser().resolve()
            print(f"[Dataset] 正在扫描: {path.name}")
            
            with h5py.File(path, "r") as f:
                total_t = f["state/gamma_liquid"].shape[0]

                ds_T = f["state/T"] if "T" in f["state"] else None
                
                for t_start in tqdm(range(0, total_t, window_size), leave=False, desc="Scanning Windows"):
                    t_end = min(total_t, t_start + window_size)

                    # ============== A. 统计温度场 ==============
                    if "T" in self.fields and ds_T is not None:
                        T_win = ds_T[t_start:t_end, :, 0].reshape(-1)
                        stats["T"]["sum"] += np.sum(T_win, dtype=np.float64)
                        stats["T"]["sq_sum"] += np.sum(T_win ** 2, dtype=np.float64)
                        stats["T"]["count"] += T_win.size

                    # ============== B. 统计速度场 (全区域 + 死区截断 + 可选变换) ==============
                    for field_name in ["Ux", "Uy", "Uz"]:
                        if field_name in self.fields and f"state/{field_name}" in f:
                            ds_vel = f[f"state/{field_name}"]
                            vel_valid = ds_vel[t_start:t_end, :, 0].reshape(-1)

                            vel_valid[np.abs(vel_valid) < velocity_dead_zone] = 0.0

                            if self.velocity_transformode == "log":
                                vel_valid = np.sign(vel_valid) * np.log1p(np.abs(vel_valid) / velocity_dead_zone)
                            elif self.velocity_transformode == "arcsinh":
                                arcsinh_scale = 1.0
                                vel_valid = np.arcsinh(vel_valid / arcsinh_scale)

                            stats[field_name]["sum"] += np.sum(vel_valid, dtype=np.float64)
                            stats[field_name]["sq_sum"] += np.sum(vel_valid ** 2, dtype=np.float64)
                            stats[field_name]["count"] += vel_valid.size

        # 4. 计算最终均值和方差，生成配置字典
        final_config = {}
        for field in self.fields:
            # 组分场由于本身就是 0-1 之间，直接分配均值 0，方差 1
            if field in ["alpha.air", "alpha.titanium", "gamma_liquid"]:
                final_config[field] = (0.0, 1.0)
            elif stats[field]["count"] > 0:
                data = stats[field]
                mean = data["sum"] / data["count"]
                std = np.sqrt(max(0, data["sq_sum"] / data["count"] - mean**2))
                final_config[field] = (float(mean), float(std))
            else:
                print(f"[Dataset 警告] 字段 {field} 在采样中未找到有效数据！将使用默认参数 (0, 1)")
                final_config[field] = (0.0, 1.0)

        # 5. 保存到 JSON 缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(final_config, f, indent=4)
            
        print(f"[Dataset] ✅ 统计完成！结果已缓存至: {cache_path}")
        for k, v in final_config.items():
            print(f"  {k:<15}: Mean = {v[0]:.4e}, Std = {v[1]:.4e}")
            
        return final_config
    
    def check(self):
        """
        检查所有文件的 condition 是否正常。
        只打印第一次出现的 condition 组合及其对应的文件名。
        """
        seen_conditions = set()
        
        print(f"\n{'='*20} Dataset Condition Check Report {'='*20}")
        print(f"Total files to check: {len(self.file_paths)}")

        for file_path in self.file_paths:
            if file_path not in self.meta_cache:
                continue

            conditions = self.meta_cache[file_path]['conditions']
            # mat_conditions = self.meta_cache[file_path]['mat_conditions'].flatten()

            if torch.is_tensor(conditions):
                cond_vals = conditions.cpu().numpy().flatten()
            else:
                cond_vals = np.array(conditions).flatten()
                
            mat_conditions = cond_vals[5:-1]

            # 2. 生成指纹 (Tuple) 用于去重
            cond_key = tuple(np.round(mat_conditions, 8))


            if cond_key not in seen_conditions:
                seen_conditions.add(cond_key)
                
                file_name = Path(file_path).name
                print(f"\n[New Condition Detected] File: {file_name}")
                
                # 根据 _process_condition_normalize 的逻辑进行切片展示
                if len(cond_vals) > 6:
                    thermal_part = cond_vals[:5]
                    material_part = cond_vals[5:-1]
                    dump_part = cond_vals[-1]
                    
                    print(f"  > Material (Norm): shape={material_part.shape}, mean={material_part.mean():.4f}")
                    print(f"    Values: {material_part}") 
                    print(f"    Values: {mat_conditions}") 
                    print(f"  > Dump ID:         {dump_part:.1f}")
                else:
                    print(f"  > Conditions: {cond_vals}")

        print(f"\n{'='*20} Check Finished. Found {len(seen_conditions)} unique condition sets. {'='*20}\n")
