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

from .utils import ChannelNormalizer, ChannelNormalizerWithLog


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

    for k in range(gz):
        idx = k * stride_z
        pt = f["point"][idx]
        z_val = pt[2]

        dist = abs(z_val - target_z)
        if dist < min_dist:
            min_dist = dist
            best_k = k

    return best_k


def _compute_2d_indices(grid_shape: Tuple[int, int, int], stride: Tuple[int, int, int], z_layer_idx: int):
    """生成特定 Z 层的二维索引。"""
    gx, gy, gz = grid_shape
    sx, sy, sz = stride
    _ = gz, sz

    xs = list(range(0, gx, sx))
    ys = list(range(0, gy, sy))
    if xs[-1] != gx - 1:
        xs.append(gx - 1)
    if ys[-1] != gy - 1:
        ys.append(gy - 1)

    ds_shape = (len(xs), len(ys))
    indices = []
    base_z = z_layer_idx * gx * gy
    for y in ys:
        base_y = base_z + y * gx
        for x in xs:
            indices.append(base_y + x)

    return np.asarray(indices, dtype=np.int32), ds_shape


def _build_2d_edges(ds_shape: Tuple[int, int], sample_ratio=1.0) -> torch.Tensor:
    """基于二维规则网格生成四向邻接边。"""
    nx, ny = ds_shape
    edges = []

    def idx(x, y):
        return x + nx * y

    for y in range(ny):
        for x in range(nx):
            cur = idx(x, y)
            if x + 1 < nx:
                edges.append((cur, idx(x + 1, y)))
            if y + 1 < ny:
                edges.append((cur, idx(x, y + 1)))

    edges_arr = np.asarray(edges, dtype=np.int32)
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
            if x in (0, nx - 1) or y in (0, ny - 1):
                node_types[x + nx * y, 0] = 1
    return torch.from_numpy(node_types)


def _process_condition_normalize(f: h5py.File, mat_mean_and_std=None):
    """对参数进行标准化操作，数据集中不变的参数暂时不传入模型"""
    cond_list = []
    thermal_cond_list = [
        ("parameter/thermal", 3, np.arange(100, 300, 10).mean(), np.arange(100, 300, 10).std()),
        ("parameter/thermal", 4, np.arange(35e-6, 45e-6, 1e-6).mean(), np.arange(35e-6, 45e-6, 1e-6).std()),
        ("parameter/thermal", 5, np.arange(2e-4, 7e-4, 1e-4).mean(), np.arange(2e-4, 7e-4, 1e-4).std()),
        ("parameter/thermal", 7, np.arange(0.35, 0.45, 0.01).mean(), np.arange(0.35, 0.45, 0.01).std()),
        ("parameter/thermal", 8, np.arange(0.2, 0.4, 0.01).mean(), np.arange(0.2, 0.4, 0.01).std()),
    ]

    for path, idx, mean_val, std_val in thermal_cond_list:
        if path in f:
            data_arr = f[path][:]
            val = data_arr.reshape(-1)[idx]
            val_norm = (float(val) - mean_val) / (std_val + 1e-6)
            cond_list.append([val_norm])
        else:
            cond_list.append([0.0])

    mat_ori_data = np.array(
        [
            [8e-07, 7400, 7.45e6, 3090, 0.05593, 1658, 1723, 684, 9.248, 0.01571, 18, 2.155e5, 3.27, 4.48, 0.36],
            [8.7e-7, 8040, 6.4e6, 3188, 0.05975, 1578, 1623, 638, 4.5847, 0.0163, 30.078, 2.61e5, 2.88, 5.26, 0.36],
            [1.33e-5, 1.67, 1000, 1000, 0.04, 1941, 1942, 520, 0.02, 0, 0.02, 0, 3.27, 4.48, 0.4],
            [5.185e-7, 8100, 4.38e6, 2835, 0.0677, 1123, 1273, 416, 63, 0, 63, 1.6344e5, 0.35, 6.97, 0.36],
        ],
        dtype=np.float32,
    )
    _ = mat_ori_data

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
        if fname in f["boundcond"]:
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


class AeroGtoDataset2D(Dataset):
    """针对 LPBF 数据的 2D 切片数据集。"""

    def __init__(
        self, config,
        mode: str = "train",
        slice_z: float = 5e-4,
        mat_data=None,
    ):
        super().__init__()
        self.config = config
        assert mode in {"train", "test"}, "mode 只能为 train 或 test"
        self.mode = mode
        self.fields = config["fields"]
        self.input_steps = config["input_steps"]
        self.horizon = config[f"horizon_{mode}"]
        self.time_stride = config.get("time_stride", 1)
        self.spatial_stride = _normalize_stride(config.get("spatial_stride", 1))
        self.normalize = config.get("normalize", True)
        self.mat_mean_and_std = mat_data
        self.samples_per_file = config.get("samples_per_file", 32)
        self.slice_z = slice_z
        self.log_velocity = config.get("log_velocity", False)

        if self.log_velocity:
            channel_class = ChannelNormalizerWithLog
        else:
            channel_class = ChannelNormalizer

        self.file_paths = _read_file_list(config[f"{self.mode}_list"])
        self.cache_path = self._generate_cache_path(config["norm_cache"])

        self.meta_cache = {}
        self.sample_keys = []
        self.max_start_per_file = []

        for file_id, path in enumerate(self.file_paths):
            meta = self._build_meta(path)
            self.meta_cache[path] = meta
            self.max_start_per_file.append(meta["max_start"])

            if mode == "train":
                for _ in range(self.samples_per_file):
                    self.sample_keys.append((file_id, None))
            else:
                step = max(1, self.horizon // 2)
                for start in range(1, meta["max_start"] + 1, step):
                    self.sample_keys.append((file_id, start))

        example_meta = next(iter(self.meta_cache.values()))
        self.cond_dim = example_meta["conditions"].shape[-1]
        self.node_num = example_meta["node_pos"].shape[0]
        self.dt = example_meta["dt"]

        num_channels = len(self.fields)
        if self.normalize and self.mode == "train":
            self.normalizer = self._load_normalizer()
        else:
            self.normalizer = channel_class(np.zeros(num_channels, dtype=np.float32), np.ones(num_channels, dtype=np.float32))

    def _generate_cache_path(self, base_dir) -> Path:
        if base_dir is None:
            base_dir = "../result/norm_cache"
        base_path = Path(base_dir).expanduser().resolve()

        if base_path.suffix == ".json":
            return base_path
        if self.file_paths:
            prefix = Path(self.file_paths[0]).stem
        else:
            prefix = "lpbf_dataset_2d"

        fields_str = "-".join(self.fields)
        path_str = "".join(sorted(self.file_paths)).encode("utf-8")
        file_hash = hashlib.md5(path_str).hexdigest()[:6]
        z_str = f"{self.slice_z:.2e}".replace("+", "")
        cache_filename = f"{prefix}_{fields_str}_z{z_str}_{file_hash}_stats_2d.json"
        
        return base_path / cache_filename

    def _load_normalizer(self):
        log_fields = ["Ux", "Uy", "Uz"]
        log_flags = [f in log_fields for f in self.fields]
        log_scale = 1e-3

        field_stats_config = {}
        if self.cache_path.is_file() and self.cache_path.exists():
            print(f"[Dataset2D] 命中归一化缓存配置: {self.cache_path.name}")
            with open(self.cache_path, "r") as f:
                loaded_config = json.load(f)
            field_stats_config.update(loaded_config)
        else:
            print(f"[Dataset2D] 未找到缓存 {self.cache_path.name}，开始自动计算...")
            computed_config = self._compute_and_cache_dataset_stats(
                cache_path=self.cache_path,
                max_sample_files=5,
                window_size=5,
                velocity_dead_zone=log_scale,
                t_threshold=301.0,
            )
            field_stats_config.update(computed_config)

        mean_list, std_list = [], []
        for fname in self.fields:
            m, s = field_stats_config.get(fname, (0.0, 1.0))
            mean_list.append(m)
            std_list.append(s)

        mean_arr = np.array(mean_list, dtype=np.float32)
        std_arr = np.array(std_list, dtype=np.float32)

        if self.log_velocity:
            return ChannelNormalizerWithLog(mean_arr, std_arr, log_flags=log_flags, log_scale=log_scale)
        return ChannelNormalizer(mean_arr, std_arr)

    def scale_2d_pos(self, node_pos):
        xx = node_pos[..., 0]
        yy = node_pos[..., 1]

        x_norm = (xx - xx.min()) / (xx.max() - xx.min())
        y_norm = (yy - yy.min()) / (yy.max() - yy.min())

        node_pos_new = torch.stack((x_norm, y_norm), dim=-1)
        return node_pos_new

    def _build_meta(self, path: str):
        path = str(Path(path).expanduser().resolve())
        with h5py.File(path, "r") as f:
            block = f["mesh/block"][0].astype(int)
            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)
            z_layer_idx = _find_closest_z_layer(f, grid_shape, self.slice_z)
            indices, ds_shape = _compute_2d_indices(grid_shape, self.spatial_stride, z_layer_idx)

            point_all = f["point"][:]
            point = point_all[indices]
            point_2d = point[:, :2]
            node_pos = torch.from_numpy(point_2d.astype(np.float32))

            edges = _build_2d_edges(ds_shape)
            node_type = _build_2d_node_type(ds_shape)
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
        if "alpha.air" not in self.fields:
            return state

        alpha_idx = self.fields.index("alpha.air")
        alpha_val = state[..., alpha_idx]
        gas_mask = alpha_val > 0.5
        masked_state = state.clone()
        for i, field in enumerate(self.fields):
            if field in ["Ux", "Uy", "Uz", "gamma_liquid"]:
                masked_state[..., i][gas_mask] = 0.0
        return masked_state

    def _load_data_with_aux(self, path: str, indices: np.ndarray, start: int):
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

            if "state/T" in f:
                T_vals = f["state/T"][time_idx][:, indices, 0]
            else:
                T_vals = np.zeros((len(time_idx), len(indices)), dtype=np.float32)

            if "state/alpha.air" in f:
                alpha_vals = f["state/alpha.air"][time_idx][:, indices, 0]
            else:
                alpha_vals = np.zeros((len(time_idx), len(indices)), dtype=np.float32)

            if "state/gamma_liquid" in f:
                gamma_vals = f["state/gamma_liquid"][time_idx][:, indices, 0]
            else:
                gamma_vals = np.zeros((len(time_idx), len(indices)), dtype=np.float32)

        return state_np, time_all, T_vals, alpha_vals, gamma_vals

    def __getitem__(self, idx):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])

        state_np, time_seq, T_np, alpha_np, gamma_np = self._load_data_with_aux(path, meta["indices"], start_idx)

        state = torch.from_numpy(state_np)
        T_t = torch.from_numpy(T_np)
        alpha_t = torch.from_numpy(alpha_np)
        gamma_t = torch.from_numpy(gamma_np)

        node_y = meta["node_pos"][:, 1]
        y_cutoff_mask = node_y > 1e-4

        gas_mask = alpha_t > 0.6
        
        if self.config.get("mask", False): # 物理场裁剪
            for i, field in enumerate(self.fields):
                if field in ["Ux", "Uy", "Uz"]:
                    state[..., i][gas_mask] = 0.0
                if field == "gamma_liquid":
                    state[..., i][:, y_cutoff_mask] = 0.0

        velocity_dead_zone = 1e-3
        for i, field in enumerate(self.fields):
            if field in ["Ux", "Uy", "Uz"]:
                vel = state[..., i]
                state[..., i] = torch.where(vel.abs() < velocity_dead_zone, torch.zeros_like(vel), vel)

        gamma_t[:, y_cutoff_mask] = 0.0

        cnum = len(self.fields)
        loss_weight = torch.ones((state.shape[0], state.shape[1], cnum), dtype=torch.float32)

        temp_mask = T_t > 301
        melt_pool_mask = (gamma_t > 0.49) & (alpha_t < 0.51)
        metal_mask = (alpha_t > 0.49) & (alpha_t < 0.55)

        for i, field in enumerate(self.fields):
            if field == "T":
                loss_weight[..., i] = 1e-4
                loss_weight[..., i][temp_mask] = 1.0
            elif field in ["Ux", "Uy", "Uz"]:
                loss_weight[..., i] = 1e-4
                loss_weight[..., i][melt_pool_mask] = 1.0
            elif field in ["alpha.air"]:
                loss_weight[..., i] = 1e-4
                loss_weight[..., i][metal_mask] = 1.0

        if self.normalize:
            state = self.normalizer.normalize(state)
            node_pos = self.scale_2d_pos(meta["node_pos"])
        else:
            node_pos = meta["node_pos"]

        rel_time = time_seq[1:] - time_seq[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)
        sample = {
            "dt": meta["dt"],
            "state": state,
            "time_seq": time_tensor,
            "node_pos": node_pos,
            "edges": meta["edges"],
            "node_type": meta["node_type"],
            "conditions": meta["conditions"],
            "loss_weight": loss_weight,
        }
        return sample

    def _compute_and_cache_dataset_stats(
        self,
        cache_path: Path,
        max_sample_files=5,
        window_size=5,
        velocity_dead_zone=1e-3,
        t_threshold=301.0,
    ):
        """内置统计计算：未命中缓存时自动采样并写入。"""
        print("\n[Dataset2D] 未找到归一化缓存，准备自动从数据集中采样计算...")

        paths = self.file_paths
        sampled_paths = random.sample(paths, min(len(paths), max_sample_files))
        print(f"[Dataset2D] 数据集总文件数: {len(paths)}, 采样计算数: {len(sampled_paths)}")

        stats = {field: {"sum": 0.0, "sq_sum": 0.0, "count": 0} for field in self.fields}

        for path_str in sampled_paths:
            path = Path(path_str).expanduser().resolve()
            print(f"[Dataset2D] 正在扫描: {path.name}")

            meta = self.meta_cache[str(path)]
            indices = meta["indices"]

            with h5py.File(path, "r") as f:
                if "state/gamma_liquid" in f:
                    total_t = f["state/gamma_liquid"].shape[0]
                else:
                    total_t = f[f"state/{self.fields[0]}"].shape[0]

                ds_T = f["state/T"] if "state/T" in f else None
                ds_gamma = f["state/gamma_liquid"] if "state/gamma_liquid" in f else None
                ds_alpha = f["state/alpha.air"] if "state/alpha.air" in f else None

                for t_start in tqdm(range(0, total_t, window_size), leave=False, desc="Scanning Windows"):
                    t_end = min(total_t, t_start + window_size)

                    if ds_gamma is not None and ds_alpha is not None:
                        gamma_win = ds_gamma[t_start:t_end, indices, 0]
                        alpha_win = ds_alpha[t_start:t_end, indices, 0]
                    else:
                        gamma_win = np.zeros((t_end - t_start, len(indices)), dtype=np.float32)
                        alpha_win = np.zeros((t_end - t_start, len(indices)), dtype=np.float32)

                    if "T" in self.fields and ds_T is not None:
                        T_win = ds_T[t_start:t_end, indices, 0]
                        T_valid = T_win[T_win > t_threshold]
                        if T_valid.size > 0:
                            stats["T"]["sum"] += np.sum(T_valid, dtype=np.float64)
                            stats["T"]["sq_sum"] += np.sum(T_valid ** 2, dtype=np.float64)
                            stats["T"]["count"] += T_valid.size

                    window_mask = np.any((gamma_win > 0.49) & (alpha_win < 0.51), axis=0)
                    if np.sum(window_mask) > 0:
                        for field_name in ["Ux", "Uy", "Uz"]:
                            if field_name in self.fields and f"state/{field_name}" in f:
                                ds_vel = f[f"state/{field_name}"]
                                vel_valid = ds_vel[t_start:t_end, indices, 0][:, window_mask].reshape(-1)
                                vel_valid[np.abs(vel_valid) < velocity_dead_zone] = 0.0

                                if self.log_velocity:
                                    vel_valid = np.sign(vel_valid) * np.log1p(np.abs(vel_valid) / velocity_dead_zone)

                                stats[field_name]["sum"] += np.sum(vel_valid, dtype=np.float64)
                                stats[field_name]["sq_sum"] += np.sum(vel_valid ** 2, dtype=np.float64)
                                stats[field_name]["count"] += vel_valid.size

        final_config = {}
        for field in self.fields:
            if field in ["alpha.air", "alpha.titanium", "gamma_liquid"]:
                final_config[field] = (0.0, 1.0)
            elif stats[field]["count"] > 0:
                data = stats[field]
                mean = data["sum"] / data["count"]
                std = np.sqrt(max(0, data["sq_sum"] / data["count"] - mean**2))
                final_config[field] = (float(mean), float(std))
            else:
                print(f"[Dataset2D 警告] 字段 {field} 在采样中未找到有效数据！将使用默认参数 (0, 1)")
                final_config[field] = (0.0, 1.0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(final_config, f, indent=4)

        print(f"[Dataset2D] ✅ 统计完成！结果已缓存至: {cache_path}")
        for k, v in final_config.items():
            print(f"  {k:<15}: Mean = {v[0]:.4e}, Std = {v[1]:.4e}")

        return final_config

    def check(self):
        """检查所有文件的 condition 是否正常。"""
        seen_conditions = set()

        print(f"\n{'=' * 20} Dataset2D Condition Check Report {'=' * 20}")
        print(f"Total files to check: {len(self.file_paths)}")

        for file_path in self.file_paths:
            if file_path not in self.meta_cache:
                continue

            conditions = self.meta_cache[file_path]["conditions"]
            if torch.is_tensor(conditions):
                cond_vals = conditions.cpu().numpy().flatten()
            else:
                cond_vals = np.array(conditions).flatten()

            mat_conditions = cond_vals[5:-1]
            cond_key = tuple(np.round(mat_conditions, 8))

            if cond_key not in seen_conditions:
                seen_conditions.add(cond_key)
                file_name = Path(file_path).name
                print(f"\n[New Condition Detected] File: {file_name}")

                if len(cond_vals) > 6:
                    material_part = cond_vals[5:-1]
                    dump_part = cond_vals[-1]
                    print(f"  > Material (Norm): shape={material_part.shape}, mean={material_part.mean():.4f}")
                    print(f"    Values: {material_part}")
                    print(f"    Values: {mat_conditions}")
                    print(f"  > Dump ID:         {dump_part:.1f}")
                else:
                    print(f"  > Conditions: {cond_vals}")

        print(f"\n{'=' * 20} Check Finished. Found {len(seen_conditions)} unique condition sets. {'=' * 20}\n")
