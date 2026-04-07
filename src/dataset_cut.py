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
    """计算下采样索引。底层 1D 索引满足 z*Nx*Ny + y*Nx + x"""
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
    """基于裁剪后的网格动态生成六向邻接边"""
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


def _build_global_node_type(grid_shape: Tuple[int, int, int], y_divide=17) -> np.ndarray:
    """生成全局网格的边界类型，避免裁剪后导致边界属性判断失真"""
    nx, ny, nz = grid_shape
    node_types = np.zeros((nz, ny, nx), dtype=np.int32)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if y == 0: node_types[z, y, x] = 1
                elif y == ny - 1: node_types[z, y, x] = 2
                else:
                    if x in (0, nx - 1) or z in (0, nz - 1):
                        if y <= y_divide: node_types[z, y, x] = 1
                        else: node_types[z, y, x] = 2
    return node_types.flatten()


def _process_condition_normalize(f: h5py.File, mat_mean_and_std=None) -> np.ndarray:
    # (此部分与你原代码保持一致)
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
        return np.array([], dtype=np.float32)

    return np.concatenate(cond_list, axis=0).astype(np.float32), (mat_mean, mat_std)


def _condition_vector(f: h5py.File, field_names: List[str]) -> np.ndarray:
    # (此部分与你原代码保持一致)
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
    return np.concatenate([thermal, material, interact, dump_mean, dump_std,
                           field_box, field_scalar, field_velocity, inicond, boundcond], axis=0).astype(np.float32)


class CutAeroGtoDataset(Dataset):
    def __init__(
        self, args,
        mode: str = "train",
        mat_data = None,
        margin: int = 4,
    ):
        super().__init__()
        data_cfg = args.data
        self.config = data_cfg
        assert mode in {"train", "test"}, "mode 只能为 train 或 test"
        self.mode = mode
        self.fields = data_cfg.get("fields", ["T"])
        self.input_steps = data_cfg.get("input_steps", 1)
        self.horizon = data_cfg.get(f"horizon_{mode}", 1)
        self.time_stride = data_cfg.get("time_stride", 1)
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
            self.normalizer = ChannelNormalizer(np.zeros(num_channels, dtype=np.float32), np.ones(num_channels, dtype=np.float32))

    def _load_normalizer(self) -> ChannelNormalizer:
        field_stats_config = {
            "T":         (5.2999e+02, 4.5454e+02),  
            "Ux":        (4.0041e-05, 2.4173e-01),
            "Uy":        (-1.6900e-05, 2.5172e-01),
            "Uz":        (3.3602e-07, 1.1976e-01),
            "alpha.air": (0, 1),   
            "alpha.titanium": (0, 1), 
            "gamma_liquid": (0, 1)}
        
        mean_list, std_list = [], []
        for fname in self.fields:
            m, s = field_stats_config[fname]
            mean_list.append(m)
            std_list.append(s)
        return ChannelNormalizer(np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32))
    
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
            block = f["mesh/block"][0].astype(int)
            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)
            
            # 计算全场下采样索引
            indices, ds_shape = _compute_downsample_indices(grid_shape, self.spatial_stride)
            nx, ny, nz = ds_shape
            
            # 读取全场坐标并重构为 3D
            point_all = f["point"][:]
            pos_min = point_all.min(axis=0).astype(np.float32)
            pos_max = point_all.max(axis=0).astype(np.float32)
            point_ds = point_all[indices]
            
            # 注意：按照你描述的规则（相邻点X变，跨NxNy Z变），reshape必须是 (nz, ny, nx)
            node_pos_3d = point_ds.reshape(nz, ny, nx, 3)
            
            # 全场类型，提取下采样点并重构 3D
            global_node_type = _build_global_node_type(grid_shape)
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
            max_start = total_steps - (self.input_steps + self.horizon * self.time_stride)

        return {
            "indices": indices,
            "ds_shape": ds_shape,
            "node_pos_3d": node_pos_3d,
            "node_type_3d": node_type_3d,
            "pos_min": pos_min,
            "pos_max": pos_max,
            "conditions": torch.from_numpy(conditions),
            "dt": dt,
            "max_start": max_start,
            "edges": np.array([[0, 1]])
        }

    def _get_dynamic_bounds(self, f: h5py.File, time_idx: np.ndarray, meta: dict):
        """核心方法：在当前时间窗口内探测活跃边界（即使 T 或 gamma 不在 self.fields 也能裁剪）"""
        indices = meta["indices"]
        nx, ny, nz = meta["ds_shape"]
        active_mask = np.zeros((nz, ny, nx), dtype=bool)

        def print_current_bounds(step_name, mask):
            z_i, y_i, x_i = np.where(mask)
            if len(x_i) == 0:
                print(f"  👉 [{step_name}] 当前活跃区全空")
            else:
                print(f"  👉 [{step_name}] X:[{x_i.min()}->{x_i.max()}] (宽{x_i.max()-x_i.min()+1}), "
                      f"Y:[{y_i.min()}->{y_i.max()}] (高{y_i.max()-y_i.min()+1}), "
                      f"Z:[{z_i.min()}->{z_i.max()}] (深{z_i.max()-z_i.min()+1})")

        # 优先嗅探温度场
        # if 'state/T' in f:
        #     T_data = f['state/T'][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)
        #     active_mask |= np.any(T_data > 1200, axis=0)
        #     # print_current_bounds("1.叠加温度场(>1200K)", active_mask)

        # 嗅探液相场
        if 'state/gamma_liquid' in f:
            gamma = f['state/gamma_liquid'][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)
            active_mask |= np.any(np.abs(gamma) > 1e-3, axis=0)
            print_current_bounds("2.叠加液相(>1e-4)", active_mask)

        # gas_mask = None
        # if 'state/alpha.air' in f:
        #     air = f['state/alpha.air'][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)
        #     gas_mask = air > 0.75

        # 嗅探速度场
        # for v in ['Ux', 'Uy', 'Uz']:
        #     if f'state/{v}' in f:
        #         V_data = f[f'state/{v}'][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)

        #         if gas_mask is not None:
        #             V_data[gas_mask] = 0.0
                    
        #         active_mask |= np.any(np.abs(V_data) > 1e-3, axis=0)

        #         print_current_bounds(f"3.叠加金属速度{v}(>1e-3)", active_mask)

        z_inds, y_inds, x_inds = np.where(active_mask)

        # 容错：若未检测到任何变化（比如刚开始冷却），默认截取中心一小块
        if len(x_inds) == 0:
            cx, cy, cz = nx // 2, ny // 2, nz // 2
            return max(0, cx-self.margin), min(nx-1, cx+self.margin), max(0, cy-self.margin), min(ny-1, cy+self.margin), max(0, cz-self.margin), min(nz-1, cz+self.margin)

        x_min_act, x_max_act = x_inds.min(), x_inds.max()
        z_min_act, z_max_act = z_inds.min(), z_inds.max()
        y_min_act = y_inds.min()

        # 处理空气边界上限
        if 'state/alpha.air' in f:
            air = f['state/alpha.air'][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)
            _, y_inds_air, _ = np.where(np.any(air < 0.75, axis=0))
            y_max_act = y_inds_air.max() if len(y_inds_air) > 0 else y_inds.max()
        else:
            y_max_act = y_inds.max()

        x_min = max(0, x_min_act - self.margin)
        x_max = min(nx - 1, x_max_act + self.margin)
        y_min = max(0, y_min_act - self.margin)
        y_max = min(ny - 1, y_max_act + self.margin)
        z_min = max(0, z_min_act - self.margin)
        z_max = min(nz - 1, z_max_act + self.margin)

        return x_min, x_max, y_min, y_max, z_min, z_max

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])

        time_idx = start_idx + np.arange(0, self.horizon + 1) * self.time_stride
        nx, ny, nz = meta["ds_shape"]
        num_channels = len(self.fields)

        with h5py.File(path, "r") as f:
            time_seq = f["time"][time_idx]
            
            # 1. 计算当前特定时间窗口的 Bounding Box (从原文件主动读取判断条件)
            x_min, x_max, y_min, y_max, z_min, z_max = self._get_dynamic_bounds(f, time_idx, meta)
            
            # 2. 读取模型真正需要的 fields 数据并直接装入 3D 张量
            channels_data = []
            for fname in self.fields:
                d = f[f"state/{fname}"][time_idx][:, meta["indices"], 0]
                channels_data.append(d)
            state_3d = np.stack(channels_data, axis=-1).reshape(-1, nz, ny, nx, num_channels)

        # 3. 按求得的局部边界进行切片
        crop_state = state_3d[:, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1, :]
        crop_pos = meta["node_pos_3d"][z_min:z_max+1, y_min:y_max+1, x_min:x_max+1, :]
        crop_type = meta["node_type_3d"][z_min:z_max+1, y_min:y_max+1, x_min:x_max+1, :]

        # 4. 根据新尺寸展平重组
        nx_new = x_max - x_min + 1
        ny_new = y_max - y_min + 1
        nz_new = z_max - z_min + 1
        N_new = nx_new * ny_new * nz_new

        state = torch.from_numpy(crop_state).reshape(-1, N_new, num_channels)
        node_pos = torch.from_numpy(crop_pos).reshape(N_new, 3)
        node_type = torch.from_numpy(crop_type).reshape(N_new, 1)

        active_mask = build_active_mask(state, self.fields, self.mask_cfg)

        # 5. 归一化 (位置依然使用全场极值)
        if self.normalize:
            state = self.normalizer.normalize(state)
            pos_min_t = torch.from_numpy(meta["pos_min"]) # 全场极值
            pos_max_t = torch.from_numpy(meta["pos_max"])

            pos_min_r = node_pos.min(dim=0).values # 相对极值
            pos_max_r = node_pos.max(dim=0).values

            node_pos = (node_pos - pos_min_r) / (pos_max_r - pos_min_r + 1e-8)

        # 6. 生成新图边
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
            "dt": meta['dt'] * self.time_stride / self.time_ref,
            "state": state,
            "time_seq": time_tensor / self.time_ref,
            "node_pos": node_pos,
            "edges": edges,
            "node_type": node_type,
            "conditions": meta["conditions"],
            "ds_shape": [nx, ny, nz],
            "cut_shape": [nx_new, ny_new, nz_new],
            "grid_shape": torch.Tensor([nx_new, ny_new, nz_new]),
            "spatial_inform": spatial_inform,
        }
        if active_mask is not None:
            sample["active_mask"] = active_mask
        return sample

# from utils import load_json_config
# import time
# config_path = r"/home/ubuntu/PhysGTO/Moving_History/config/aerogto_cut_version_easypool.json"
# args = load_json_config(config_path)
# data_cfg = args.data

# t0 = time.time()

# test_dataset = CutAeroGtoDataset(
#         data_cfg=data_cfg,
#         file_list=data_cfg["train_list"],
#         mode="train",
#         fields=data_cfg.get("fields", ["T"]),
#         input_steps=data_cfg.get("input_steps", 1),
#         horizon=data_cfg.get("horizon_train", 1),
#         time_stride=data_cfg.get("time_stride", 1),
#         spatial_stride=data_cfg.get("spatial_stride", 1),
#         normalize=data_cfg.get("normalize", True),
#         samples_per_file=data_cfg.get("samples_per_file", 32),
#         norm_cache=data_cfg.get("norm_cache"),
#     )

# print(f"✅ Dataset 初始化完成！耗时: {time.time() - t0:.3f} 秒。共生成 {len(test_dataset)} 个样本。")

# num_test_samples = min(5, len(test_dataset))

# fetch_times = []
# for i in range(num_test_samples):
#     # 随机挑几个索引
#     idx = torch.randint(0, len(test_dataset), (1,)).item()
    
#     t_start = time.perf_counter()
#     sample = test_dataset[idx]
#     t_end = time.perf_counter()
    
#     fetch_times.append(t_end - t_start)
    
#     # 验证裁剪出的节点数 N
#     N = sample['node_pos'].shape[0]
#     E = sample['edges'].shape[0]
#     nx, ny, nz = sample["ds_shape"]
#     nx_new, ny_new, nz_new = sample["cut_shape"]
    
#     print(f"\n▶ 样本 {idx}:")
#     print(f"  - 耗时: {fetch_times[-1]*1000:.2f} ms")
#     print(f"  - 场景形状由{nx}*{ny}*{nz} -> {nx_new}*{ny_new}*{nz_new}")
#     print(f"  - 动态节点数 (N): {N}, 动态边数 (E): {E}")
#     print(f"  - state 形状: {sample['state'].shape}   (期望: [6, {N}, 5])")
#     print(f"  - node_pos 形状: {sample['node_pos'].shape} (期望: [{N}, 3])")
#     print(f"  - node_type 形状: {sample['node_type'].shape} (期望: [{N}, 1])")
#     print(f"  - edges 形状: {sample['edges'].shape}    (期望: [{E}, 2])")

# print(f"\n⚡ __getitem__ 平均拉取耗时: {sum(fetch_times)/len(fetch_times)*1000:.2f} ms")