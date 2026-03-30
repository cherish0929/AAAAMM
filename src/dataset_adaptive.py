"""
dataset_adaptive.py
===================
Dataset for the adaptive-graph training pipeline.

Key difference from AeroGtoDataset / CutAeroGtoDataset:
  - Pre-computes and caches the **backbone** sub-graph (coarse, fixed).
  - Also provides the **full-resolution** node positions and state so that
    the adaptive graph manager can dynamically select refinement points
    during autoregressive rollout.
  - The full-res grid shape is included in each sample so edge builders
    can be called on-the-fly.

The backbone graph (indices, positions, edges) is built once during
`_build_meta` and reused for every sample from the same file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import ChannelNormalizer
from .adaptive_graph import build_backbone_indices, build_backbone_edges


# ---------------------------------------------------------------------------
# Helpers reused from existing dataset modules
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
        raise ValueError("No valid data file paths found")
    return paths


def _normalize_stride(stride) -> Tuple[int, int, int]:
    if isinstance(stride, int):
        return (stride, stride, stride)
    if isinstance(stride, (list, tuple)) and len(stride) == 3:
        return tuple(int(x) for x in stride)
    raise ValueError("spatial_stride must be int or length-3 list/tuple")


def _compute_downsample_indices(grid_shape, stride):
    gx, gy, gz = grid_shape
    sx, sy, sz = stride
    xs = list(range(0, gx, sx))
    ys = list(range(0, gy, sy))
    zs = list(range(0, gz, sz))
    if xs[-1] != gx - 1: xs.append(gx - 1)
    if ys[-1] != gy - 1: ys.append(gy - 1)
    if zs[-1] != gz - 1: zs.append(gz - 1)
    ds_shape = (len(xs), len(ys), len(zs))
    indices = []
    for z in zs:
        for y in ys:
            base = z * gx * gy + y * gx
            for x in xs:
                indices.append(base + x)
    return np.asarray(indices, dtype=np.int64), ds_shape


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
            val = f[path][:].reshape(-1)[idx]
            cond_list.append([(float(val) - mean_val) / (std_val + 1e-6)])
        else:
            cond_list.append([0.0])
    mat_all = f["parameter/material"][:]
    mat_2 = mat_all[0:-1, :]
    if mat_mean_and_std is None:
        mat_mean, mat_std = np.mean(mat_2, axis=0), np.std(mat_2, axis=0)
    else:
        mat_mean, mat_std = mat_mean_and_std
    mat_norm = (mat_2 - mat_mean) / (mat_std + 1e-6)
    cond_list.append(mat_norm.reshape(-1))
    dump = len(np.unique(f["parameter/dump"][:][:, -1]))
    cond_list.append([dump])
    return np.concatenate(cond_list, axis=0).astype(np.float32), (mat_mean, mat_std)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AdaptiveGraphDataset(Dataset):
    """Dataset that provides full-resolution data + pre-built backbone graph.

    Each sample contains:
        - state:           [1 + horizon, N_full, C]  full-resolution states
        - time_seq:        [horizon, 1]
        - fullres_pos:     [N_full, 3]               positions (normalized)
        - fullres_pos_raw: [N_full, 3]               positions (un-normalized, for writeback)
        - backbone_edges:  [E_bb, 2]                 backbone edge index (local)
        - backbone_indices:[N_bb]                     indices in full grid
        - backbone_shape:  [3]                        (nx_b, ny_b, nz_b)
        - fullres_shape:   [3]                        (nx, ny, nz)
        - conditions:      [cond_dim]
        - dt:              scalar
    """

    def __init__(
        self,
        data_cfg: dict,
        file_list: Iterable[str],
        mode: str = "train",
        fields: List[str] = ["T"],
        input_steps: int = 1,
        horizon: int = 5,
        time_stride: int = 1,
        spatial_stride: Union[int, Tuple[int, int, int]] = 1,
        normalize: bool = True,
        samples_per_file: int = 32,
        norm_cache: Optional[str] = None,
        mat_data=None,
        adaptive_cfg: Optional[dict] = None,
    ):
        super().__init__()
        assert mode in {"train", "test"}
        self.config = data_cfg
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

        # adaptive-specific config
        self.adaptive_cfg = adaptive_cfg or {}
        bb_cfg = self.adaptive_cfg.get("backbone", {})
        self.backbone_stride = _normalize_stride(bb_cfg.get("stride", [2, 2, 2]))
        self.bb_edge_sample_ratio = bb_cfg.get("edge_sample_ratio", 1.0)

        self.file_paths = _read_file_list(file_list)
        self.meta_cache = {}
        self.sample_keys = []

        for file_id, path in enumerate(self.file_paths):
            meta = self._build_meta(path)
            self.meta_cache[path] = meta
            if mode == "train":
                for _ in range(samples_per_file):
                    self.sample_keys.append((file_id, None))
            else:
                step = max(1, horizon // 2)
                for start in range(1, meta["max_start"] + 1, step):
                    self.sample_keys.append((file_id, start))

        example_meta = next(iter(self.meta_cache.values()))
        self.cond_dim = example_meta["conditions"].shape[-1]
        self.node_num = example_meta["n_full"]
        self.dt = example_meta["dt"]

        if self.normalize and self.mode == "train":
            self.normalizer = self._load_normalizer()
        else:
            n_ch = len(self.fields)
            self.normalizer = ChannelNormalizer(
                np.zeros(n_ch, dtype=np.float32),
                np.ones(n_ch, dtype=np.float32),
            )

    # ---- normalizer -------------------------------------------------------

    def _load_normalizer(self) -> ChannelNormalizer:
        field_stats = {
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
            m, s = field_stats[fname]
            mean_list.append(m)
            std_list.append(s)
        return ChannelNormalizer(
            np.array(mean_list, dtype=np.float32),
            np.array(std_list, dtype=np.float32),
        )

    # ---- metadata / backbone pre-computation ------------------------------

    def scale_3D_pos(self, node_pos: torch.Tensor) -> torch.Tensor:
        xx, yy, zz = node_pos[..., 0], node_pos[..., 1], node_pos[..., 2]
        x_norm = (xx - xx.min()) / (xx.max() - xx.min() + 1e-8)
        y_norm = (yy - yy.min()) / (yy.max() - yy.min() + 1e-8)
        z_norm = (zz - zz.min()) / (zz.max() - zz.min() + 1e-8)
        return torch.stack((x_norm, y_norm, z_norm), dim=-1)

    def _build_meta(self, path: str):
        path = str(Path(path).expanduser().resolve())
        with h5py.File(path, "r") as f:
            block = f["mesh/block"][0].astype(int)
            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)

            # full-resolution indices (with dataset-level spatial_stride)
            fullres_indices, fullres_ds_shape = _compute_downsample_indices(
                grid_shape, self.spatial_stride
            )
            n_full = len(fullres_indices)

            # full-res positions
            point_all = f["point"][:]
            fullres_pos = torch.from_numpy(
                point_all[fullres_indices].astype(np.float32)
            )

            # backbone: further downsample from fullres grid
            bb_indices, bb_shape = build_backbone_indices(
                fullres_ds_shape, self.backbone_stride
            )
            # bb_indices are indices into the *fullres* grid (not the original grid)
            bb_edges = build_backbone_edges(bb_shape, self.bb_edge_sample_ratio)

            # conditions
            if self.normalize:
                if self.mat_mean_and_std is None:
                    conditions, self.mat_mean_and_std = _process_condition_normalize(f)
                else:
                    conditions, _ = _process_condition_normalize(f, self.mat_mean_and_std)
            else:
                from .dataset import _condition_vector
                conditions = _condition_vector(f, self.fields)
            conditions = torch.from_numpy(conditions)

            time_all = f["time"][:]
            dt = np.float32(np.mean(np.diff(time_all)))
            total_steps = len(time_all)
            max_start = total_steps - (self.input_steps + self.horizon * self.time_stride)

        return {
            "grid_shape": grid_shape,
            "fullres_indices": fullres_indices,
            "fullres_ds_shape": fullres_ds_shape,
            "n_full": n_full,
            "fullres_pos": fullres_pos,
            "backbone_indices": bb_indices,       # indices into fullres grid
            "backbone_shape": bb_shape,
            "backbone_edges": bb_edges,
            "conditions": conditions,
            "dt": dt,
            "max_start": max_start,
        }

    # ---- data loading -----------------------------------------------------

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])

        # read full-resolution state for the time window
        time_idx = start_idx + np.arange(0, self.horizon + 1) * self.time_stride
        with h5py.File(path, "r") as f:
            time_seq = f["time"][time_idx]
            channels = []
            for fname in self.fields:
                d = f[f"state/{fname}"][time_idx][:, meta["fullres_indices"], 0]
                channels.append(d)
        state_np = np.stack(channels, axis=-1).astype(np.float32)  # [T, N_full, C]
        state = torch.from_numpy(state_np)

        fullres_pos_raw = meta["fullres_pos"].clone()

        if self.normalize:
            state = self.normalizer.normalize(state)
            fullres_pos = self.scale_3D_pos(meta["fullres_pos"])
        else:
            fullres_pos = meta["fullres_pos"]

        rel_time = time_seq[1:] - time_seq[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)

        nx_f, ny_f, nz_f = meta["fullres_ds_shape"]

        return {
            "dt": meta["dt"],
            "state": state,                                                   # [1+H, N_full, C]
            "time_seq": time_tensor,                                          # [H, 1]
            "fullres_pos": fullres_pos,                                       # [N_full, 3] normalized
            "fullres_pos_raw": fullres_pos_raw,                               # [N_full, 3] raw
            "fullres_shape": torch.tensor([nx_f, ny_f, nz_f], dtype=torch.long),
            "backbone_indices": torch.from_numpy(meta["backbone_indices"]),   # [N_bb] in fullres
            "backbone_shape": torch.tensor(meta["backbone_shape"], dtype=torch.long),
            "backbone_edges": meta["backbone_edges"],                         # [E_bb, 2]
            "conditions": meta["conditions"],
        }
