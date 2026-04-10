# -*- coding: utf-8 -*-
import os, random
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import imageio
from tqdm import tqdm
from pathlib import Path
import h5py
from torch.amp import GradScaler, autocast # 引入 AMP 模块

# 引入项目模块
from src.dataset_fast import AeroGtoDataset
from src.utils import load_json_config, set_seed


def _build_model(model_cfg, cond_dim, default_dt, device):
    """根据 config 中的 model.name 动态构建对应模型（兼容所有变体）"""
    model_name = model_cfg.get("name", "PhysGTO")

    if model_name == "PhysGTO":
        from src.physgto import Model
    elif model_name == "PhysGTO_v2":
        from src.physgto_v2 import Model
    elif model_name == "gto_res":
        from src.physgto_res import Model
    elif model_name == "gto_lnn":
        from src.gto_lnn import Model
    elif model_name == "gto_attnres_multi":
        from src.physgto_attnres_multi import Model
    elif model_name == "gto_attnres_multi_v2":
        from src.physgto_attnres_multi_v2 import Model
    elif model_name == "gto_res_attnres":
        from src.physgto_res_attnres import Model
    elif model_name == "gto_attnres_multi_v3":
        from src.physgto_attnres_multi_v3 import Model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    kwargs = dict(
        space_size=model_cfg.get("space_size", 3),
        pos_enc_dim=model_cfg.get("pos_enc_dim", 5),
        cond_dim=cond_dim,
        N_block=model_cfg.get("N_block", 4),
        in_dim=model_cfg.get("in_dim", 4),
        out_dim=model_cfg.get("out_dim", 4),
        enc_dim=model_cfg.get("enc_dim", 128),
        n_head=model_cfg.get("n_head", 4),
        n_token=model_cfg.get("n_token", 64),
        dt=model_cfg.get("dt", default_dt),
    )

    # AttnRes 系列需要额外参数
    if model_name in ("gto_attnres_multi", "gto_attnres_multi_v2", "gto_res_attnres", "gto_attnres_multi_v3"):
        kwargs["n_fields"] = model_cfg.get("n_fields", model_cfg.get("in_dim", 2))
        kwargs["cross_attn_heads"] = model_cfg.get("cross_attn_heads", 4)

    if model_name in ("gto_attnres_multi_v2", "gto_res_attnres"):
        kwargs["attn_res_mode"] = model_cfg.get("attn_res_mode", "block_inter")

    if model_name in ("PhysGTO_v2", "gto_attnres_multi_v3"):
        kwargs["spatial_dim"] = model_cfg.get("spatial_dim", 10)
        kwargs["pos_x_boost"] = model_cfg.get("pos_x_boost", 2)

    if model_name == "gto_attnres_multi_v3":
        kwargs["n_latent"] = model_cfg.get("n_latent", 4)

    return Model(**kwargs).to(device)

class AeroGtoPredictor:
    def __init__(self, config_path, mode="test", model_path=None, device_str="cuda"):
        self.args = load_json_config(config_path)
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")

        data_cfg = self.args.data
        model_cfg = self.args.model

        self.fields = data_cfg.get("fields", ["T"])
        space_dim = model_cfg.get("space_size", 3)


        # 如果配置中有 use_cut_dataset 标记，或者你想强制启用，可以在这里修改
        # if HAS_CUT_DATASET and data_cfg.get("use_cut_dataset", False):
        #     print("[Init] Using AeroGtoDatasetCut...")
        #     DatasetClass = AeroGtoDatasetCut
        
        print("[Init] Loading Train Dataset (for Normalizer)...")
        # 即使是 inference，通常也需要 TrainSet 的统计数据来做 Normalizer
        train_dataset = AeroGtoDataset(
            args=self.args,
            mode="train")

        if mode == "test":
            self.dataset = AeroGtoDataset(
                args=self.args,
                mode="test",
                mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None
            )


            self.dataset.normalizer = train_dataset.normalizer
            self.dataset._sync_norm_cache()  # 同步 norm_mean/norm_std 缓存
        
        elif mode == "train":
            self.dataset = train_dataset

        self.fields = self.dataset.fields
        print(f"[Init] Fields to predict: {self.fields}")

        # 2. 构建模型
        print("[Init] Building Model...")
        cond_dim = self.args.model.get("cond_dim") or self.dataset.cond_dim
        default_dt = self.args.model.get("dt", self.dataset.dt)

        self.model = _build_model(model_cfg, cond_dim, default_dt, self.device)

        # 3. 加载权重（支持 EMA）
        if model_path is None:
            save_root = Path(self.args.save_path)
            model_path = save_root / "nn" / f"{self.args.name}_best.pt"

        print(f"[Init] Loading weights from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weight file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 优先使用 EMA 权重（main_v2 保存的 checkpoint 包含 ema_shadow）
        if "ema_shadow" in checkpoint:
            print("[Init] Loading EMA shadow weights.")
            ema_shadow = checkpoint["ema_shadow"]
            state_dict = self.model.state_dict()
            for name in ema_shadow:
                if name in state_dict:
                    state_dict[name] = ema_shadow[name]
            self.model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        
        self.normalizer = self.dataset.normalizer
        self.normalizer.to(self.device)

    def _load_gt_interface(self, sample_idx, interface_name="alpha.air"):
        """
        辅助函数：额外从H5文件中读取真实的界面场 (GT)，用于画图
        """
        # 1. 找到对应的文件和时间步
        file_id, start_idx = self.dataset.sample_keys[sample_idx]
        path = self.dataset.file_paths[file_id]
        meta = self.dataset.meta_cache[path]
        
        # 2. 确定时间索引
        if start_idx is None: start_idx = 1
        
        # 计算对应的时间索引序列
        time_indices = start_idx + np.arange(1, self.dataset.horizon + 1) * self.dataset.time_stride
        indices = meta["indices"] # 这里直接使用 meta 中的索引 (如果是 cut dataset，这已经是裁剪后的索引)
        
        try:
            with h5py.File(path, 'r') as f:
                if f"state/{interface_name}" not in f:
                    return None
                
                # 读取数据
                # 注意：H5读取逻辑需要与 Dataset 中的逻辑保持一致
                # 如果是 DatasetCut，indices 是针对全场的 indices，可以直接 fancy indexing
                data = f[f"state/{interface_name}"][time_indices] # [Horizon, All_Points, 1]
                data = data[:, indices, 0] # 下采样/裁剪 [Horizon, N]
                return data
        except Exception as e:
            print(f"[Warn] Failed to load GT interface '{interface_name}': {e}")
            return None

    def predict_rollout(self, sample_idx, interface_field="alpha.air"):
        """
        执行自回归预测，并准备绘图所需的所有数据
        """
        sample = self.dataset[sample_idx]
        model_name = self.args.model.get("name", "PhysGTO")

        use_amp, check_point = self.args.train.get("use_amp", False), self.args.train.get("check_point", False)
        # 增加 Batch 维度并移至 GPU
        state_seq = sample["state"].unsqueeze(0).to(self.device)
        node_pos = sample["node_pos"].unsqueeze(0).to(self.device)
        edges = sample["edges"].unsqueeze(0).to(self.device)
        time_seq = sample["time_seq"].unsqueeze(0).to(self.device)
        _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3")
        if _use_spatial:
            spatial_inform = sample["spatial_inform"].unsqueeze(0).to(self.device)
        conditions = sample["conditions"].unsqueeze(0).to(self.device).float()
        
        # [保留用户逻辑] 使用第50个样本的 conditions (可能是为了测试泛化或者固定工况)
        # cond_idx = 50 if len(self.dataset) > 50 else 0
        # conditions = self.dataset[cond_idx]["conditions"].unsqueeze(0).to(self.device).float()

        dt = sample["dt"]

        state_0 = state_seq[:, 0] 
        gt_seq = state_seq[:, 1:]

        print(f"[Predict] Running autoregressive inference...")
        with torch.no_grad():
            if use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    if _use_spatial:
                        pred_seq = self.model.autoregressive(
                            state_0, node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point=check_point)
                    else:
                        pred_seq = self.model.autoregressive(
                            state_0, node_pos, edges, time_seq, conditions, dt, check_point=check_point)
            else:
                if _use_spatial:
                    pred_seq = self.model.autoregressive(
                        state_0, node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point=check_point)
                else:
                    pred_seq = self.model.autoregressive(
                        state_0, node_pos, edges, time_seq, conditions, dt, check_point=check_point)

            pred_real = self.normalizer.denormalize(pred_seq)
            gt_real = self.normalizer.denormalize(gt_seq)

        # --- [关键修改] 获取真实物理坐标 ---
        # 通过 sample_idx 回溯 metadata，获取原始(未归一化)坐标
        file_id, _ = self.dataset.sample_keys[sample_idx]
        path = self.dataset.file_paths[file_id]
        meta = self.dataset.meta_cache[path]
        
        # meta["node_pos"] 存储的是真实的物理坐标 (对于 Cut Dataset，就是裁剪后的物理坐标)
        raw_coords = meta["node_pos"]
        if isinstance(raw_coords, torch.Tensor):
            raw_coords = raw_coords.cpu().numpy()

        # --- 边界数据获取逻辑 ---
        gt_interface = self._load_gt_interface(sample_idx, interface_field)
        
        pred_interface = None
        if interface_field in self.fields:
            idx = self.fields.index(interface_field)
            pred_interface = pred_real[0, :, :, idx].cpu().numpy() # [Horizon, N]
        else:
            print(f"[Info] Model does not predict '{interface_field}', so Pred boundary will be missing.")
        
        return {
            "pred": pred_real.cpu().numpy()[0], # [T, N, C]
            "gt": gt_real.cpu().numpy()[0],     # [T, N, C]
            "coords": raw_coords,               # [N, 3] 真实物理坐标
            "gt_interface": gt_interface,       # [T, N] or None
            "pred_interface": pred_interface    # [T, N] or None
        }
    
    def _is_phase_field(self, field_name):
        return any(k in field_name for k in ["alpha", "gamma", "frac"])

    def _build_slice_mask(self, coords, axis, slice_pos=None, min_points=32):
        axis_id = {'x': 0, 'y': 1, 'z': 2}[axis]
        axis_values = coords[:, axis_id]
        if slice_pos is None:
            slice_pos = 0.5 * (np.nanmin(axis_values) + np.nanmax(axis_values))

        axis_span = max(np.nanmax(axis_values) - np.nanmin(axis_values), 1e-12)
        thickness = max(axis_span * 2e-3, 1e-8)
        mask = np.abs(axis_values - slice_pos) <= thickness

        for _ in range(8):
            if np.sum(mask) >= min_points:
                break
            thickness *= 1.8
            mask = np.abs(axis_values - slice_pos) <= thickness

        if np.sum(mask) < min_points:
            # 如果规则切片仍然太稀疏，退化为“最近邻 slab”
            dist = np.abs(axis_values - slice_pos)
            k = min(len(dist), max(min_points, int(0.03 * len(dist))))
            idx = np.argpartition(dist, k - 1)[:k]
            mask = np.zeros_like(dist, dtype=bool)
            mask[idx] = True

        return mask, slice_pos

    def _interp_grid(self, pts_x, pts_y, vals, Xi, Yi, method="cubic"):
        if vals is None:
            return None
        points = np.column_stack([pts_x, pts_y])
        values = np.asarray(vals)

        interp_try = [method]
        if method != "linear":
            interp_try.append("linear")
        interp_try.append("nearest")

        Z = None
        for m in interp_try:
            try:
                Z = griddata(points, values, (Xi, Yi), method=m)
            except Exception:
                Z = None
            if Z is not None and not np.all(np.isnan(Z)):
                break

        if Z is None:
            return None

        if np.isnan(Z).any():
            try:
                Z_nearest = griddata(points, values, (Xi, Yi), method="nearest")
                Z = np.where(np.isnan(Z), Z_nearest, Z)
            except Exception:
                pass
        return Z

    def _smooth_for_plot(self, Z, sigma):
        if Z is None or sigma is None or sigma <= 0:
            return Z
        if np.all(np.isnan(Z)):
            return Z
        nan_mask = np.isnan(Z)
        fill_value = np.nanmedian(Z)
        if not np.isfinite(fill_value):
            fill_value = 0.0
        Z_fill = np.where(nan_mask, fill_value, Z)
        Z_smooth = gaussian_filter(Z_fill, sigma=sigma, mode="nearest")
        Z_smooth[nan_mask] = np.nan
        return Z_smooth

    def plot_slice(self, 
                   result_dict, 
                   time_step, 
                   field_name="T", 
                   axis="z", 
                   slice_pos=None, 
                   res=320, # 分辨率
                   vmin=None, vmax=None, 
                   return_array=False,
                   save_path=None,
                   interface=True,
                   smooth_sigma=0.9,
                   rel_err_cap=1.0):
        
        try:
            f_idx = self.fields.index(field_name)
        except ValueError:
            print(f"Field {field_name} not found.")
            return
        
        # 如果调用方没给范围，则尝试自动给一个默认值
        field_limits = {
            "T": (300, 3500),         # 温度：室温到沸点以上
            "alpha.air": (0, 1),      # VOF: 0-1
            "alpha.titanium": (0, 1), # VOF: 0-1
            "gamma_liquid": (0, 1),   # 液相分数: 0-1
        }

        if field_name in field_limits:
            vmin, vmax = field_limits[field_name]
            # print(f"[GIF] Using fixed limits for {field_name}: {vmin} ~ {vmax}")
        else:
            # 自动计算全局范围，防止每一帧跳变
            all_pred = result_dict["pred"][..., f_idx]
            all_gt = result_dict["gt"][..., f_idx]
            combined = np.concatenate([all_pred.reshape(-1), all_gt.reshape(-1)])
            vmin, vmax = -1.0, 1.0
            vmin = np.nanpercentile(combined, 1.0)
            vmax = np.nanpercentile(combined, 99.0)
            if np.isclose(vmin, vmax):
                vmin = min(np.nanmin(all_pred), np.nanmin(all_gt))
                vmax = max(np.nanmax(all_pred), np.nanmax(all_gt))

        pred_data = result_dict["pred"][time_step, :, f_idx]
        gt_data = result_dict["gt"][time_step, :, f_idx]
        coords = result_dict["coords"] # 使用真实坐标
        
        gt_int = None
        pred_int = None
        if result_dict.get("gt_interface") is not None:
            gt_int = result_dict["gt_interface"][time_step]
        if result_dict.get("pred_interface") is not None:
            pred_int = result_dict["pred_interface"][time_step]

        # 2. 确定切片掩码（自适应厚度 + 兜底最近邻）
        mask, slice_pos = self._build_slice_mask(coords, axis, slice_pos=slice_pos, min_points=32)
        
        # 坐标映射
        if axis == 'x':
            pts_x, pts_y = coords[mask, 1], coords[mask, 2]
            xlabel, ylabel = 'Y (m)', 'Z (m)'
        elif axis == 'y':
            pts_x, pts_y = coords[mask, 0], coords[mask, 2]
            xlabel, ylabel = 'X (m)', 'Z (m)'
        else:
            pts_x, pts_y = coords[mask, 0], coords[mask, 1]
            xlabel, ylabel = 'X (m)', 'Y (m)'

        if np.sum(mask) < 4 or len(np.unique(pts_x)) < 2 or len(np.unique(pts_y)) < 2:
            print(f"[Error] Too few points ({np.sum(mask)}) in slice {axis}={slice_pos:.3e}.")
            return None

        # 3. 插值 + 平滑
        xi = np.linspace(pts_x.min(), pts_x.max(), res)
        yi = np.linspace(pts_y.min(), pts_y.max(), res)
        Xi, Yi = np.meshgrid(xi, yi)

        is_phase = self._is_phase_field(field_name)
        interp_method = "linear" # if is_phase else "cubic"
        sigma = 0.5 if is_phase else smooth_sigma

        Zi_pred_raw = self._interp_grid(pts_x, pts_y, pred_data[mask], Xi, Yi, method=interp_method)
        Zi_gt_raw = self._interp_grid(pts_x, pts_y, gt_data[mask], Xi, Yi, method=interp_method)
        if Zi_pred_raw is None or Zi_gt_raw is None:
            print("[Error] Interpolation failed.")
            return None

        Zi_pred = self._smooth_for_plot(Zi_pred_raw, sigma=sigma)
        Zi_gt = self._smooth_for_plot(Zi_gt_raw, sigma=sigma)
        
        if is_phase:
            Zi_pred = np.clip(Zi_pred, 0, 1)
            Zi_gt = np.clip(Zi_gt, 0, 1)
        
        # 误差计算
        Zi_err = np.abs(Zi_pred - Zi_gt)
        denom_floor = max(1e-8, 1e-3 * np.nanmax(np.abs(Zi_gt)))
        Zi_rel_err = Zi_err / np.maximum(np.abs(Zi_gt), denom_floor)
        Zi_rel_err = np.clip(Zi_rel_err, 0, rel_err_cap)

        # 误差图仅做轻微平滑，目的是消除插值噪点并提升可读性
        Zi_err_vis = self._smooth_for_plot(Zi_err, sigma=min(max(sigma, 0.3), 1.0))
        Zi_rel_err_vis = self._smooth_for_plot(Zi_rel_err, sigma=min(max(sigma, 0.3), 1.0))
        
        if interface:
            Zi_gt_int = self._interp_grid(pts_x, pts_y, gt_int[mask] if gt_int is not None else None, Xi, Yi, method="linear")
            Zi_pred_int = self._interp_grid(pts_x, pts_y, pred_int[mask] if pred_int is not None else None, Xi, Yi, method="linear")
        else:
            Zi_gt_int, Zi_pred_int = None, None

        # 4. 绘图
        plt.rcParams.update({
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        })
        fig, axes = plt.subplots(1, 4, figsize=(21, 5), constrained_layout=True)
        
        # 确保 vmin/vmax 有效
        if vmin is None: vmin = np.nanmin(Zi_gt)
        if vmax is None: vmax = np.nanmax(Zi_gt)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9

        cmap = 'inferno' if 'T' in field_name else 'viridis'
        imshow_args = dict(
            extent=(xi.min(), xi.max(), yi.min(), yi.max()),
            origin='lower',
            interpolation='bicubic',
            aspect='equal'
        )

        # GT
        im0 = axes[0].imshow(Zi_gt, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_args)
        axes[0].set_title(f"Ground Truth ({field_name})")
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
        cb0.set_label(field_name)
        
        # Pred
        im1 = axes[1].imshow(Zi_pred, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_args)
        axes[1].set_title(f"Prediction ({field_name})")
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
        cb1.set_label(field_name)
        
        # Abs Error
        err_vmax = np.nanpercentile(Zi_err_vis, 99.0) if np.any(np.isfinite(Zi_err_vis)) else 1.0
        err_vmax = max(err_vmax, 1e-12)
        im2 = axes[2].imshow(Zi_err_vis, cmap='magma', vmin=0, vmax=err_vmax, **imshow_args)
        axes[2].set_title("Absolute Error")
        cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03, extend='max')
        cb2.set_label("|Pred - GT|")

        # Rel Error
        rel_vmax = min(rel_err_cap, max(0.1, np.nanpercentile(Zi_rel_err_vis, 99.0)))
        im3 = axes[3].imshow(Zi_rel_err_vis, cmap='magma', vmin=0, vmax=rel_vmax, **imshow_args)
        axes[3].set_title(f"Relative Error (clip={rel_err_cap:.2f})")
        cb3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.03, extend='max')
        cb3.set_label("|Pred-GT| / max(|GT|, eps)")

        # 画线
        lines, labels = [], []
        if Zi_gt_int is not None:
            # 画在 GT 和 Pred 上
            axes[0].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
            axes[1].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
            lines.append(plt.Line2D([0], [0], color='white', linestyle='--', linewidth=1.5))
            labels.append('GT Interface')
            
        if Zi_pred_int is not None:
             lines.append(plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5))
             labels.append('Pred Interface')

        # 画在 Error 图上
        for ax in [axes[2], axes[3]]:
            if Zi_gt_int is not None:
                ax.contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
            if Zi_pred_int is not None:
                ax.contour(Xi, Yi, Zi_pred_int, levels=[0.5], colors='red', linestyles=':', linewidths=1.5)
            
            if lines:
                ax.legend(lines, labels, loc='upper right', framealpha=0.6, fontsize='small')

        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='both')
            ax.set_aspect('equal', adjustable='box')

        if return_array:
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            buf.seek(0)
            img = imageio.v2.imread(buf)
            buf.close()
            return img
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def generate_gif(self, result_dict, field_name="T", axis="z", slice_pos=None, git_path="result/inference_results", interface=True):
        print(f"[GIF] Generating animation for field '{field_name}'...")
        
        f_idx = self.fields.index(field_name)
        
        # [修改] 统一 GIF 色标范围配置
        field_limits = {
            "T": (300, 3500),         # 温度：室温到沸点以上
            "alpha.air": (0, 1),      # VOF: 0-1
            "alpha.titanium": (0, 1), # VOF: 0-1
            "gamma_liquid": (0, 1),   # 液相分数: 0-1
        }

        if field_name in field_limits:
            vmin, vmax = field_limits[field_name]
            # print(f"[GIF] Using fixed limits for {field_name}: {vmin} ~ {vmax}")
        else:
            # 自动计算全局范围，防止每一帧跳变
            all_pred = result_dict["pred"][..., f_idx]
            all_gt = result_dict["gt"][..., f_idx]
            combined = np.concatenate([all_pred.reshape(-1), all_gt.reshape(-1)])
            vmin, vmax = -1.0, 1.0
            # vmin = np.nanpercentile(combined, 1.0)
            # vmax = np.nanpercentile(combined, 99.0)
            # if np.isclose(vmin, vmax):
            #     vmin = min(np.nanmin(all_pred), np.nanmin(all_gt))
            #     vmax = max(np.nanmax(all_pred), np.nanmax(all_gt))
            print(f"[GIF] Auto-detected global limits: {vmin:.2f} ~ {vmax:.2f}")

        frames = []
        horizon = result_dict["pred"].shape[0]

        for t in tqdm(range(horizon), desc="Rendering Frames"):
            img = self.plot_slice(
                result_dict, 
                time_step=t, 
                field_name=field_name, 
                axis=axis, 
                slice_pos=slice_pos, 
                vmin=vmin, 
                vmax=vmax, 
                return_array=True,
                interface=interface,
                res=280
            )
            if img is not None:
                frames.append(img)
        
        if len(frames) > 0:
            print("Unique frame shapes:", sorted(set([f.shape for f in frames])))
            imageio.mimsave(git_path, frames, fps=10, loop=0)
            print(f"[GIF] Saved to {git_path}")
        else:
            print("[GIF] Error: No frames generated.")

if __name__ == "__main__":
    MODE = "test"
    # === 配置区域 ===
    # CONFIG_PATH = "config/easypool/GTO_easypool_stronger.json"
    # CONFIG_PATH 也可以是 list，依次处理多个配置：
    CONFIG_PATH = [
        # "config/easypool/GTO_easypool.json",
        # "config/easypool/GTO_easypool_stronger.json",
        # "config/easypool/GTO_attnres_easypool.json",
        # "config/easypool/GTO_attnres_easypool_stronger.json",
        # "config/easypool/GTO_2_easypool_stronger.json",
        # "config/easypool/GTO_attnres_3_easypool_stronger.json"
        "config/keyhole/GTO_keyhole_stronger.json",
        "config/keyhole/GTO_attnres_keyhole_stronger.json",
        "config/keyhole/GTO_attnres_3_keyhole_stronger.json",
    ]

    FIELD_TO_PLOT = None   # None 表示所有场；或指定如 "T" / "alpha.air"
    SLICE_AXIS = "z"        # 'x', 'y', 'z'
    SLICE_POS = None
    INTERFACE_FIELD = "alpha.air"
    NUM_SAMPLES = 2  # 每个 config 随机推理的样本数
    sample_idxs = [207, 229]

    cfg_list = CONFIG_PATH if isinstance(CONFIG_PATH, list) else [CONFIG_PATH]

    for cfg_path in cfg_list:
        try:
            predictor = AeroGtoPredictor(cfg_path, MODE)
            if FIELD_TO_PLOT is None:
                OUT_DIR = f"result_keyhole/inference_standard/inference/{predictor.args.name}/{MODE}/batch"
            else:
                OUT_DIR = f"result_keyhole/inference_standard/inference/{predictor.args.name}/{MODE}/{FIELD_TO_PLOT}"
            os.makedirs(OUT_DIR, exist_ok=True)
        except Exception as e:
            print(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        dataset_length = len(predictor.dataset)
        print(f"Dataset size: {dataset_length}")
        # sample_idxs = random.sample(range(dataset_length), min(NUM_SAMPLES, dataset_length))

        for sample_idx in sample_idxs:
            results = predictor.predict_rollout(sample_idx=sample_idx, interface_field=INTERFACE_FIELD)

            if FIELD_TO_PLOT is None:
                for field in predictor.fields:
                    gif_path = os.path.join(OUT_DIR, f"rollout_sample{sample_idx}_{field}.gif")
                    predictor.generate_gif(
                        results, field_name=field, axis=SLICE_AXIS,
                        slice_pos=SLICE_POS, git_path=gif_path, interface=False
                    )
            else:
                gif_path = os.path.join(OUT_DIR, f"rollout_sample{sample_idx}_{FIELD_TO_PLOT}.gif")
                if os.path.exists(gif_path):
                    continue
                predictor.generate_gif(
                    results, field_name=FIELD_TO_PLOT, axis=SLICE_AXIS,
                    slice_pos=SLICE_POS, git_path=gif_path, interface=False
                )