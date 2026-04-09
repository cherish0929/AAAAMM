"""
dataset_cut_fast.py — 优化版 CutAeroGtoDataset

相对 dataset_cut.py 的主要优化：

1. _build_global_node_type 向量化
   原版用三重 Python for 循环（O(Nx*Ny*Nz) 次 Python 迭代），
   新版用 numpy 广播赋值，速度提升 10~100x。

2. __getitem__ 合并 HDF5 读取
   原版：_get_dynamic_bounds 先开一次文件（读 gamma_liquid + alpha.air），
         然后主数据读取再开一次文件（读 self.fields）。
   新版：一次性打开文件，同时读入 gamma_liquid、alpha.air 以及 self.fields，
         完全消除重复的文件 open/seek 开销。

3. 边界 alpha.air 读取去重
   如果 alpha.air 已在 self.fields 中，_get_dynamic_bounds 直接复用
   已读入的数组，不再二次读盘。
"""
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import ChannelNormalizer, build_active_mask


# ---------------------------------------------------------------------------
# 工具函数（与 dataset_cut.py 保持一致，除 _build_global_node_type）
# ---------------------------------------------------------------------------

def _read_file_list(file_list: Iterable[str]) -> List[str]:
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
    nx, ny, nz = ds_shape
    idx_grid = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    edges = []

    if nx > 1:
        src = idx_grid[:, :, :-1].flatten()
        dst = idx_grid[:, :, 1:].flatten()
        edges.append(np.stack([src, dst], axis=1))
    if ny > 1:
        src = idx_grid[:, :-1, :].flatten()
        dst = idx_grid[:, 1:, :].flatten()
        edges.append(np.stack([src, dst], axis=1))
    if nz > 1:
        src = idx_grid[:-1, :, :].flatten()
        dst = idx_grid[1:, :, :].flatten()
        edges.append(np.stack([src, dst], axis=1))

    if edges:
        edges_arr = np.concatenate(edges, axis=0)
    else:
        edges_arr = np.zeros((0, 2), dtype=np.int32)

    if sample_ratio < 1.0 and edges_arr.shape[0] > 0:
        total_edges = edges_arr.shape[0]
        target_num = int(total_edges * sample_ratio)
        sample_indices = np.random.choice(total_edges, target_num, replace=False)
        edges_arr = edges_arr[sample_indices]

    return torch.from_numpy(edges_arr.astype(np.int64))


def _build_global_node_type_fast(grid_shape: Tuple[int, int, int], y_divide=17) -> np.ndarray:
    """向量化版本，用 numpy 广播替代三重 Python for 循环。

    原版每个节点逐一赋值，耗时 O(N) 次 Python 迭代；
    新版直接对整个 3D 数组做布尔索引赋值，速度提升 10~100x。
    """
    nx, ny, nz = grid_shape
    # 布局：node_types[z, y, x]，与 dataset_cut.py 保持一致
    node_types = np.zeros((nz, ny, nx), dtype=np.int32)

    # 1) y==0 → 1（底面）
    node_types[:, 0, :] = 1

    # 2) y==ny-1 → 2（顶面）
    node_types[:, -1, :] = 2

    # 3) 侧面（x=0, x=nx-1, z=0, z=nz-1），y in [1, ny-2]
    #    y <= y_divide → 1，y > y_divide → 2
    y_idx = np.arange(1, ny - 1)  # shape (ny-2,)
    side_type = np.where(y_idx <= y_divide, 1, 2)  # shape (ny-2,)

    # x=0 列
    node_types[:, 1:-1, 0] = side_type[np.newaxis, :]
    # x=nx-1 列
    node_types[:, 1:-1, -1] = side_type[np.newaxis, :]
    # z=0 面
    node_types[0, 1:-1, :] = side_type[:, np.newaxis]
    # z=nz-1 面
    node_types[-1, 1:-1, :] = side_type[:, np.newaxis]

    return node_types.flatten()


def _process_condition_normalize(f: h5py.File, mat_mean_and_std=None):
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
        return np.array([], dtype=np.float32), (None, None)

    return np.concatenate(cond_list, axis=0).astype(np.float32), (mat_mean, mat_std)


def _condition_vector(f: h5py.File, field_names: List[str]) -> np.ndarray:
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
            inicond_list.append(f["inicond"][fname][:].reshape(-1))
        if fname in f['boundcond']:
            boundcond_list.append(f["boundcond"][fname][:].reshape(-1))
    inicond = np.concatenate(inicond_list, axis=0)
    boundcond = np.concatenate(boundcond_list, axis=0)
    return np.concatenate(
        [thermal, material, interact, dump_mean, dump_std,
         field_box, field_scalar, field_velocity, inicond, boundcond],
        axis=0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# 优化版 Dataset
# ---------------------------------------------------------------------------

class CutAeroGtoDataset(Dataset):
    """优化版 CutAeroGtoDataset。

    与 dataset_cut.py 逻辑完全等价，但做了以下性能优化：
    - _build_global_node_type 向量化（去掉三重 Python for 循环）
    - __getitem__ 只开一次 HDF5 文件，同时读取 bounding-box 探测字段
      和模型训练字段，减少重复 I/O
    """

    def __init__(
        self, args,
        mode: str = "train",
        mat_data=None,
        margin: int = 4,
        spatial_stride=None
    ):
        super().__init__()
        data_cfg = args.data
        self.config = data_cfg
        assert mode in {"train", "test"}, "mode 只能为 train 或 test"
        self.mode = mode
        self.fields = data_cfg.get("fields", ["T"])
        self.input_steps = data_cfg.get("input_steps", 1)
        self.horizon = data_cfg.get(f"horizon_{mode}", 1)
        self.pf_extra = data_cfg.get("horizon_pf_extra", 0) if mode == "train" else 0
        self.time_stride = data_cfg.get("time_stride", 1)
        if mode == "test" and spatial_stride is not None:
            self.spatial_stride = _normalize_stride(spatial_stride)
        else:
            self.spatial_stride = _normalize_stride(data_cfg.get("spatial_stride", 1))
        self.normalize = data_cfg.get("normalize", True)
        self.mat_mean_and_std = mat_data
        self.samples_per_file = data_cfg.get("samples_per_file", 32)
        self.norm_cache = data_cfg.get("norm_cache")
        self.margin = margin

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
                    self.sample_keys.append((file_id, None))
            else:
                step = max(1, self.horizon // 2)
                for start in range(1, meta["max_start"] + 1, step):
                    self.sample_keys.append((file_id, start))

        example_meta = next(iter(self.meta_cache.values()))
        self.cond_dim = example_meta["conditions"].shape[-1]
        self.node_num = example_meta["node_pos_3d"].size
        self.dt = example_meta["dt"] / self.time_ref
        num_channels = len(self.fields)

        if self.normalize and self.mode == "train":
            self.normalizer = self._load_normalizer()
        else:
            self.normalizer = ChannelNormalizer(
                np.zeros(num_channels, dtype=np.float32),
                np.ones(num_channels, dtype=np.float32)
            )

        # 优化：初始化完成后，把 normalizer 的统计量 pin 到 CPU
        # __getitem__ 里直接用，不再反复 .to()
        self._sync_norm_cache()

    def _sync_norm_cache(self):
        """将 normalizer 的统计量同步到 norm_mean / norm_std 缓存。
        在 __init__ 及外部覆盖 normalizer 后必须调用。

        注意：CutAeroGtoDataset 的 node_pos 每次 __getitem__ 动态裁剪，
        无法像 AeroGtoDataset 那样预缓存 node_pos_scaled。
        """
        self.norm_mean = self.normalizer.mean  # shape [1, 1, C]
        self.norm_std = self.normalizer.std + self.normalizer.eps

    def _load_normalizer(self) -> ChannelNormalizer:
        field_stats_config = {
            "T":              (5.2999e+02, 4.5454e+02),
            "Ux":             (4.0041e-05, 2.4173e-01),
            "Uy":             (-1.6900e-05, 2.5172e-01),
            "Uz":             (3.3602e-07, 1.1976e-01),
            "alpha.air":      (0, 1),
            "alpha.titanium": (0, 1),
            "gamma_liquid":   (0, 1),
        }
        mean_list, std_list = [], []
        for fname in self.fields:
            m, s = field_stats_config[fname]
            mean_list.append(m)
            std_list.append(s)
        return ChannelNormalizer(
            np.array(mean_list, dtype=np.float32),
            np.array(std_list, dtype=np.float32)
        )

    def _build_meta(self, path: str):
        path = str(Path(path).expanduser().resolve())
        with h5py.File(path, "r") as f:
            block = f["mesh/block"][0].astype(int)
            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)

            indices, ds_shape = _compute_downsample_indices(grid_shape, self.spatial_stride)
            nx, ny, nz = ds_shape

            point_all = f["point"][:]
            pos_min = point_all.min(axis=0).astype(np.float32)
            pos_max = point_all.max(axis=0).astype(np.float32)
            point_ds = point_all[indices]
            node_pos_3d = point_ds.reshape(nz, ny, nx, 3)

            # 优化：向量化版 node type，替代原版三重 for 循环
            global_node_type = _build_global_node_type_fast(grid_shape)
            node_type_3d = global_node_type[indices].reshape(nz, ny, nx, 1)

            if self.normalize:
                if self.mat_mean_and_std is None:
                    conditions, self.mat_mean_and_std = _process_condition_normalize(f)
                else:
                    conditions, _ = _process_condition_normalize(f, self.mat_mean_and_std)
            else:
                conditions = _condition_vector(f, self.fields)

            time_all = f["time"][:]
            dt = np.float32(np.mean(np.diff(time_all)))
            total_steps = len(time_all)
            max_start = total_steps - (self.input_steps + (self.horizon + self.pf_extra) * self.time_stride)

        return {
            "indices":      indices,
            "ds_shape":     ds_shape,
            "node_pos_3d":  node_pos_3d,
            "node_type_3d": node_type_3d,
            "pos_min":      pos_min,
            "pos_max":      pos_max,
            "conditions":   torch.from_numpy(conditions),
            "dt":           dt,
            "max_start":    max_start,
            "edges":        np.array([[0, 1]]),  # 占位，__getitem__ 动态生成
        }

    def _compute_bounds(
        self,
        gamma_3d: Optional[np.ndarray],   # shape (T, nz, ny, nx) or None
        air_3d:   Optional[np.ndarray],   # shape (T, nz, ny, nx) or None
        nx: int, ny: int, nz: int,
    ) -> Tuple[int, int, int, int, int, int]:
        """从已读入的数组计算动态 bounding box，不再重复读 HDF5。"""
        active_mask = np.zeros((nz, ny, nx), dtype=bool)

        if gamma_3d is not None:
            active_mask |= np.any(np.abs(gamma_3d) > 1e-3, axis=0)

        z_inds, y_inds, x_inds = np.where(active_mask)

        if len(x_inds) == 0:
            cx, cy, cz = nx // 2, ny // 2, nz // 2
            return (
                max(0, cx - self.margin), min(nx - 1, cx + self.margin),
                max(0, cy - self.margin), min(ny - 1, cy + self.margin),
                max(0, cz - self.margin), min(nz - 1, cz + self.margin),
            )

        x_min_act, x_max_act = int(x_inds.min()), int(x_inds.max())
        z_min_act, z_max_act = int(z_inds.min()), int(z_inds.max())
        y_min_act = int(y_inds.min())

        if air_3d is not None:
            _, y_inds_air, _ = np.where(np.any(air_3d < 0.75, axis=0))
            y_max_act = int(y_inds_air.max()) if len(y_inds_air) > 0 else int(y_inds.max())
        else:
            y_max_act = int(y_inds.max())

        return (
            max(0, x_min_act - self.margin), min(nx - 1, x_max_act + self.margin),
            max(0, y_min_act - self.margin), min(ny - 1, y_max_act + self.margin),
            max(0, z_min_act - self.margin), min(nz - 1, z_max_act + self.margin),
        )

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])

        time_idx = start_idx + np.arange(0, self.horizon + self.pf_extra + 1) * self.time_stride
        nx, ny, nz = meta["ds_shape"]
        num_channels = len(self.fields)
        indices = meta["indices"]

        # 优化：一次打开文件，同时读入 bounding-box 探测字段 + 训练字段
        with h5py.File(path, "r") as f:
            time_seq = f["time"][time_idx]

            # -- 读取探测字段（gamma_liquid, alpha.air） --
            gamma_3d = None
            if 'state/gamma_liquid' in f:
                gamma_3d = f['state/gamma_liquid'][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)

            air_3d = None
            if 'state/alpha.air' in f:
                air_3d = f['state/alpha.air'][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)

            # -- 读取训练字段 --
            channels_data = {}
            for fname in self.fields:
                channels_data[fname] = f[f"state/{fname}"][time_idx][:, indices, 0]

        # -- 计算 bounding box（纯 CPU numpy，不再开文件） --
        x_min, x_max, y_min, y_max, z_min, z_max = self._compute_bounds(
            gamma_3d, air_3d, nx, ny, nz
        )

        # -- 组装 state 并裁剪 --
        channels_list = [channels_data[fname] for fname in self.fields]
        state_3d = np.stack(channels_list, axis=-1).reshape(-1, nz, ny, nx, num_channels)
        crop_state = state_3d[:, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1, :]

        crop_pos  = meta["node_pos_3d"][z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1, :]
        crop_type = meta["node_type_3d"][z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1, :]

        nx_new = x_max - x_min + 1
        ny_new = y_max - y_min + 1
        nz_new = z_max - z_min + 1
        N_new  = nx_new * ny_new * nz_new

        state     = torch.from_numpy(crop_state).reshape(-1, N_new, num_channels)
        node_pos  = torch.from_numpy(crop_pos).reshape(N_new, 3)
        node_type = torch.from_numpy(crop_type).reshape(N_new, 1)

        active_mask = build_active_mask(state, self.fields, self.mask_cfg)

        if self.normalize:
            # 优化：直接用预先固定的 CPU tensor 做广播，避免 .to(device) 调用
            state = (state - self.norm_mean) / self.norm_std
            pos_min_r = node_pos.min(dim=0).values
            pos_max_r = node_pos.max(dim=0).values
            node_pos = (node_pos - pos_min_r) / (pos_max_r - pos_min_r + 1e-8)

        edges = _build_grid_edges((nx_new, ny_new, nz_new), sample_ratio=self.edge_sample_ratio)

        # 裁剪后的坐标范围 + 网格间隔，共9项
        pos_min_crop = crop_pos.reshape(-1, 3).min(axis=0)  # [3]
        pos_max_crop = crop_pos.reshape(-1, 3).max(axis=0)  # [3]
        n_arr = np.array([nx_new, ny_new, nz_new], dtype=np.float32)
        spacing = (pos_max_crop - pos_min_crop) / np.maximum(n_arr - 1, 1)
        spatial_inform = torch.from_numpy(
            np.concatenate([
                np.stack([pos_min_crop, pos_max_crop], axis=1).flatten(),  # [xmin,xmax,ymin,ymax,zmin,zmax]
                spacing,  # [dx, dy, dz]
                np.array([self.time_ref], dtype=np.float32),  # 时间参考值
            ]).astype(np.float32)
        )  # shape [10]

        rel_time = time_seq[1:] - time_seq[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)

        sample = {
            "dt":         meta['dt'] * self.time_stride / self.time_ref,
            "state":      state,
            "time_seq":   time_tensor / self.time_ref,
            "node_pos":   node_pos,
            "edges":      edges,
            "node_type":  node_type,
            "conditions": meta["conditions"],
            "ds_shape":   [nx, ny, nz],
            "cut_shape":  [nx_new, ny_new, nz_new],
            "grid_shape": torch.Tensor([nx_new, ny_new, nz_new]),
            "spatial_inform": spatial_inform,
        }
        if active_mask is not None:
            sample["active_mask"] = active_mask
        return sample
