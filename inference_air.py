# -*- coding: utf-8 -*-
"""
inference_air.py — 针对体积分数 (VOF) 场的专用推理脚本
============================================================
自动检测配置中所有体积分数场 (alpha.*, gamma.*, *frac*)，
支持单物理场和多物理场配置，逐 VOF 场分别计算指标和生成可视化。

核心关注点：界面 (alpha ≈ 0.5) 的预测精度，而非全场数值。

可视化内容 (每个 VOF 场独立生成):
  1. 界面对比图：差异发散色图 + GT/Pred 0.5 等值线 + 分歧区域高亮
  2. GT / Pred VOF 场填充等值线 (并排对比)
  3. 界面附近误差热图 (仅展示界面区域的绝对误差)
  4. 打印每个样本的 loss / L2 / RMSE / IoU / Dice 等指标
============================================================
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import imageio
from tqdm import tqdm
from pathlib import Path
from torch.amp import autocast

from src.dataset_fast import AeroGtoDataset
from src.utils import load_json_config


# ────────────────────────────────────────────
#  Metrics (复用 train.py 的相对 L2 / RMSE)
# ────────────────────────────────────────────

def _relative_l2(pred, target):
    error = pred - target
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-6)
    return torch.mean(norm_error, dim=-1).mean(dim=-1)  # [B]


def _rmse(pred, target):
    diff = pred - target
    mse = torch.mean(diff ** 2)
    return torch.sqrt(mse)


def _each_step_l2(pred, target):
    """按时间步计算相对 L2, 返回 [T]"""
    T = target.shape[1]
    losses = torch.zeros(T, device=pred.device)
    for t in range(T):
        error = pred[:, t] - target[:, t]
        norm_error = torch.norm(error, dim=-2) / (torch.norm(target[:, t], dim=-2) + 1e-6)
        losses[t] = norm_error.mean()
    return losses


def _interface_iou(pred_np, gt_np, threshold=0.5):
    """
    界面 IoU: 将 alpha 二值化后计算交并比。
    IoU 衡量"气相区域"重合度，与界面位置直接相关。
    """
    pred_bin = (pred_np >= threshold).astype(bool)
    gt_bin = (gt_np >= threshold).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return intersection / union


def _interface_dice(pred_np, gt_np, threshold=0.5):
    """Dice coefficient"""
    pred_bin = (pred_np >= threshold).astype(bool)
    gt_bin = (gt_np >= threshold).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def _interface_band_mae(pred_np, gt_np, band_lo=0.2, band_hi=0.8):
    """
    仅在界面带 (band_lo < gt < band_hi) 内的 MAE.
    这直接衡量界面附近的预测误差。
    """
    mask = (gt_np >= band_lo) & (gt_np <= band_hi)
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(pred_np[mask] - gt_np[mask]))


# ────────────────────────────────────────────
#  Model builder (与 inference_v1 相同)
# ────────────────────────────────────────────

def _build_model(model_cfg, cond_dim, default_dt, device):
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


# ════════════════════════════════════════════
#  AirFieldPredictor
# ════════════════════════════════════════════

class AirFieldPredictor:
    """专门针对 alpha.air 等 VOF 场的推理和可视化工具"""

    def __init__(self, config_path, mode="test", model_path=None, device_str="cuda"):
        self.args = load_json_config(config_path)
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.device}")

        data_cfg = self.args.data
        model_cfg = self.args.model

        # 1. 数据集
        print("[Init] Loading Train Dataset (for Normalizer)...")
        train_dataset = AeroGtoDataset(args=self.args, mode="train")

        if mode == "test":
            self.dataset = AeroGtoDataset(
                args=self.args, mode="test",
                mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None
            )
            self.dataset.normalizer = train_dataset.normalizer
            self.dataset._sync_norm_cache()
        else:
            self.dataset = train_dataset

        self.fields = self.dataset.fields
        print(f"[Init] Fields: {self.fields}")

        # 自动检测所有体积分数场 (alpha.*, gamma.*, *frac*)
        self.vof_fields = self._find_vof_fields()
        if not self.vof_fields:
            print("[Warn] No VOF field detected; falling back to index 0")
            self.vof_fields = [(self.fields[0], 0)]
        print(f"[Init] VOF fields for analysis: {self.vof_fields}")

        # 2. 模型
        print("[Init] Building Model...")
        cond_dim = self.args.model.get("cond_dim") or self.dataset.cond_dim
        default_dt = self.args.model.get("dt", self.dataset.dt)
        self.model = _build_model(model_cfg, cond_dim, default_dt, self.device)

        # 3. 权重
        if model_path is None:
            save_root = Path(self.args.save_path)
            model_path = save_root / "nn" / f"{self.args.name}_best.pt"

        print(f"[Init] Loading weights: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weight file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "ema_shadow" in checkpoint:
            print("[Init] Using EMA shadow weights.")
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

    # ──────────────────────────────
    #  VOF field detection
    # ──────────────────────────────

    def _find_vof_fields(self):
        """Return list of (field_name, channel_index) for all volume-fraction fields."""
        vof_keywords = ("alpha", "gamma", "frac")
        result = []
        for i, fname in enumerate(self.fields):
            if any(kw in fname.lower() for kw in vof_keywords):
                result.append((fname, i))
        return result

    # ──────────────────────────────
    #  推理 + 指标计算
    # ──────────────────────────────

    def predict_and_evaluate(self, sample_idx):
        """
        对单个样本执行推理，返回 result dict 和 metrics dict。
        result["pred"] / result["gt"] 为 [T, N, C] 全通道数据。
        metrics["per_field"] 为每个 VOF 场的详细指标。
        """
        sample = self.dataset[sample_idx]
        model_name = self.args.model.get("name", "PhysGTO")
        use_amp = self.args.train.get("use_amp", False)
        check_point = self.args.train.get("check_point", False)

        state_seq = sample["state"].unsqueeze(0).to(self.device)
        node_pos = sample["node_pos"].unsqueeze(0).to(self.device)
        edges = sample["edges"].unsqueeze(0).to(self.device)
        time_seq = sample["time_seq"].unsqueeze(0).to(self.device)
        conditions = sample["conditions"].unsqueeze(0).to(self.device).float()
        _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3")
        if _use_spatial:
            spatial_inform = sample["spatial_inform"].unsqueeze(0).to(self.device)
        dt = sample["dt"]

        state_0 = state_seq[:, 0]
        gt_seq = state_seq[:, 1:]  # [1, T, N, C]

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

        # ---- 全局指标 ----
        metrics = {}
        metrics["MSE_normalized"] = torch.mean((pred_seq - gt_seq) ** 2).item()

        # ---- 逐 VOF 场指标 ----
        per_field = {}
        for field_name, idx in self.vof_fields:
            fm = {}
            pred_ch = pred_real[..., idx:idx + 1]  # [1, T, N, 1]
            gt_ch = gt_real[..., idx:idx + 1]

            fm["relative_L2"] = _relative_l2(pred_ch, gt_ch).item()
            fm["RMSE"] = _rmse(pred_ch, gt_ch).item()
            fm["each_step_L2"] = _each_step_l2(pred_ch, gt_ch).cpu().numpy()

            pred_np = pred_real[0, :, :, idx].cpu().numpy()  # [T, N]
            gt_np = gt_real[0, :, :, idx].cpu().numpy()

            T = pred_np.shape[0]
            ious, dices, band_maes = [], [], []
            for t in range(T):
                ious.append(_interface_iou(pred_np[t], gt_np[t]))
                dices.append(_interface_dice(pred_np[t], gt_np[t]))
                band_maes.append(_interface_band_mae(pred_np[t], gt_np[t]))

            fm["IoU_per_step"] = np.array(ious)
            fm["Dice_per_step"] = np.array(dices)
            fm["band_MAE_per_step"] = np.array(band_maes)
            fm["mean_IoU"] = np.mean(ious)
            fm["mean_Dice"] = np.mean(dices)
            fm["mean_band_MAE"] = np.mean(band_maes)

            per_field[field_name] = fm

        metrics["per_field"] = per_field

        # 获取物理坐标
        file_id, _ = self.dataset.sample_keys[sample_idx]
        path = self.dataset.file_paths[file_id]
        meta = self.dataset.meta_cache[path]
        raw_coords = meta["node_pos"]
        if isinstance(raw_coords, torch.Tensor):
            raw_coords = raw_coords.cpu().numpy()

        result = {
            "pred": pred_real[0].cpu().numpy(),   # [T, N, C] (全通道 denormalized)
            "gt": gt_real[0].cpu().numpy(),       # [T, N, C]
            "coords": raw_coords,                 # [N, 3]
        }

        return result, metrics

    def print_metrics(self, sample_idx, metrics, out_dir=None):
        """打印单个样本的详细指标，并可选同步写入 out_dir/metrics_sample{idx}.txt"""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  Sample {sample_idx} -- Metrics Summary")
        lines.append(f"{'='*60}")
        lines.append(f"  MSE (norm):        {metrics['MSE_normalized']:.4e}")

        for field_name, fm in metrics["per_field"].items():
            lines.append(f"\n  --- {field_name} ---")
            lines.append(f"  Relative L2:       {fm['relative_L2']:.4e}")
            lines.append(f"  RMSE:              {fm['RMSE']:.4e}")
            lines.append(f"  Mean IoU:          {fm['mean_IoU']:.4f}")
            lines.append(f"  Mean Dice:         {fm['mean_Dice']:.4f}")
            lines.append(f"  Mean Band MAE:     {fm['mean_band_MAE']:.4e}")
            lines.append(f"  Per-step L2:       {[f'{v:.4e}' for v in fm['each_step_L2']]}")
            lines.append(f"  Per-step IoU:      {[f'{v:.4f}' for v in fm['IoU_per_step']]}")
            lines.append(f"  Per-step Dice:     {[f'{v:.4f}' for v in fm['Dice_per_step']]}")
            lines.append(f"  Per-step BandMAE:  {[f'{v:.4e}' for v in fm['band_MAE_per_step']]}")
        lines.append(f"{'='*60}")

        text = "\n".join(lines)
        print(text)
        if out_dir is not None:
            txt_path = os.path.join(out_dir, f"metrics_sample{sample_idx}.txt")
            with open(txt_path, "w") as f:
                f.write(text + "\n")
            print(f"[Saved] {txt_path}")

    # ──────────────────────────────
    #  切片辅助
    # ──────────────────────────────

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
            dist = np.abs(axis_values - slice_pos)
            k = min(len(dist), max(min_points, int(0.03 * len(dist))))
            idx = np.argpartition(dist, k - 1)[:k]
            mask = np.zeros_like(dist, dtype=bool)
            mask[idx] = True

        return mask, slice_pos

    def _interp_grid(self, pts_x, pts_y, vals, Xi, Yi, method="linear"):
        if vals is None:
            return None
        points = np.column_stack([pts_x, pts_y])
        values = np.asarray(vals)

        Z = None
        for m in [method, "linear", "nearest"]:
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
                Z_near = griddata(points, values, (Xi, Yi), method="nearest")
                Z = np.where(np.isnan(Z), Z_near, Z)
            except Exception:
                pass
        return Z

    def _smooth(self, Z, sigma):
        if Z is None or sigma is None or sigma <= 0:
            return Z
        if np.all(np.isnan(Z)):
            return Z
        nan_mask = np.isnan(Z)
        fill = np.nanmedian(Z)
        if not np.isfinite(fill):
            fill = 0.0
        Z_fill = np.where(nan_mask, fill, Z)
        Z_s = gaussian_filter(Z_fill, sigma=sigma, mode="nearest")
        Z_s[nan_mask] = np.nan
        return Z_s

    # ──────────────────────────────
    #  核心可视化: 界面对比
    # ──────────────────────────────

    def _contour_outlined(self, ax, X, Y, Z, levels, color, lw, ls='-', zorder=5):
        """绘制带黑色描边的等值线，提升在任何背景上的可见度。"""
        ax.contour(X, Y, Z, levels=levels,
                   colors='black', linewidths=lw + 1.5, linestyles=ls, zorder=zorder)
        return ax.contour(X, Y, Z, levels=levels,
                          colors=color, linewidths=lw, linestyles=ls, zorder=zorder + 1)

    def plot_interface(self, result, time_step, field_name=None, field_idx=None,
                       axis="z", slice_pos=None,
                       res=320, save_path=None, return_array=False,
                       smooth_sigma=0.5, metrics=None):
        """
        绘制 4 个子图 (针对指定的 VOF 场):
          1. 界面对比: 差异发散色图 (固定 ±0.1) + GT/Pred 0.5 等值线 + 分歧区域高亮
          2. GT alpha 填充 + 0.5 等值线
          3. Pred alpha 填充 + GT 0.5 等值线叠加对比
          4. 界面附近的误差热图 (仅 0.1 < alpha < 0.9 区域)
          底部: 当前步指标 (上) + 全局均值指标 (下)
        """
        # 默认选第一个 VOF 场
        if field_name is None:
            field_name, field_idx = self.vof_fields[0]
        elif field_idx is None:
            field_idx = self.fields.index(field_name)

        coords = result["coords"]
        pred_data = result["pred"][time_step, :, field_idx]  # [N]
        gt_data = result["gt"][time_step, :, field_idx]

        # 切片
        mask, slice_pos = self._build_slice_mask(coords, axis, slice_pos)

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
            print(f"[Error] Too few points ({np.sum(mask)}) for slice")
            return None

        xi = np.linspace(pts_x.min(), pts_x.max(), res)
        yi = np.linspace(pts_y.min(), pts_y.max(), res)
        Xi, Yi = np.meshgrid(xi, yi)

        Zi_gt_raw = self._interp_grid(pts_x, pts_y, gt_data[mask], Xi, Yi, method="linear")
        Zi_pred_raw = self._interp_grid(pts_x, pts_y, pred_data[mask], Xi, Yi, method="linear")
        if Zi_gt_raw is None or Zi_pred_raw is None:
            print("[Error] Interpolation failed.")
            return None

        Zi_gt = np.clip(self._smooth(Zi_gt_raw, sigma=smooth_sigma), 0, 1)
        Zi_pred = np.clip(self._smooth(Zi_pred_raw, sigma=smooth_sigma), 0, 1)

        extent = (xi.min(), xi.max(), yi.min(), yi.max())
        imshow_args = dict(extent=extent, origin='lower', aspect='equal')

        # ──────── 绘图 ────────
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.facecolor": "#f8f8f8",
            "figure.facecolor": "white",
        })

        # 构建 metrics 文本 (当前步在上，全局均值在下)
        metrics_text_step = None
        metrics_text_mean = None
        if metrics is not None and field_name in metrics.get("per_field", {}):
            fm = metrics["per_field"][field_name]
            step_l2 = fm["each_step_L2"][time_step] if time_step < len(fm["each_step_L2"]) else float('nan')
            step_iou = fm["IoU_per_step"][time_step] if time_step < len(fm["IoU_per_step"]) else float('nan')
            step_dice = fm["Dice_per_step"][time_step] if time_step < len(fm["Dice_per_step"]) else float('nan')
            step_bmae = fm["band_MAE_per_step"][time_step] if time_step < len(fm["band_MAE_per_step"]) else float('nan')
            metrics_text_step = (
                f"Step {time_step}   "
                f"L2: {step_l2:.4e}    "
                f"IoU: {step_iou:.4f}    "
                f"Dice: {step_dice:.4f}    "
                f"Band MAE: {step_bmae:.4e}"
            )
            metrics_text_mean = (
                f"Overall   "
                f"MSE(norm): {metrics['MSE_normalized']:.4e}    "
                f"Rel L2: {fm['relative_L2']:.4e}    "
                f"RMSE: {fm['RMSE']:.4e}    "
                f"Mean IoU: {fm['mean_IoU']:.4f}    "
                f"Mean Dice: {fm['mean_Dice']:.4f}    "
                f"Mean Band MAE: {fm['mean_band_MAE']:.4e}"
            )

        # 使用 gridspec 布局: 上面 4 个子图, 下方留文本区域
        has_metrics = metrics_text_step is not None
        fig_h = 8.5 if has_metrics else 6
        fig = plt.figure(figsize=(26, fig_h))
        if has_metrics:
            gs = fig.add_gridspec(2, 4, height_ratios=[6, 1.2], hspace=0.12)
        else:
            gs = fig.add_gridspec(1, 4)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        # ─── Panel 1: 界面对比图 (核心面板) ───
        diff_field = Zi_pred - Zi_gt
        # 固定色彩范围 ±0.1，超出部分 clamp 到边界色
        im0 = axes[0].imshow(diff_field, cmap='RdBu_r', vmin=-0.1, vmax=0.1,
                             interpolation='bicubic', **imshow_args)

        # 分歧区域高亮: GT 和 Pred 在 0.5 阈值上分类不一致的区域
        gt_above = Zi_gt >= 0.5
        pred_above = Zi_pred >= 0.5
        disagree = (gt_above != pred_above).astype(float)
        disagree_masked = np.where(disagree > 0.5, 1.0, np.nan)
        axes[0].imshow(disagree_masked, cmap='Oranges', vmin=0, vmax=2,
                       alpha=0.45, interpolation='nearest', **imshow_args)

        # GT 等值线 (0.5) — 黄色实线 (带黑色描边)
        self._contour_outlined(axes[0], Xi, Yi, Zi_gt, [0.5],
                               color='#FFD700', lw=1.8, ls='-')
        # Pred 等值线 (0.5) — 品红虚线 (带黑色描边)
        self._contour_outlined(axes[0], Xi, Yi, Zi_pred, [0.5],
                               color='#FF00FF', lw=1.8, ls='--')

        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, extend='both')
        cb0.set_label("Pred - GT")

        axes[0].set_title("Interface Comparison", fontweight='bold')
        legend_lines = [
            Line2D([0], [0], color='#FFD700', lw=2, ls='-',  label='GT (0.5)'),
            Line2D([0], [0], color='#FF00FF', lw=2, ls='--', label='Pred (0.5)'),
            Line2D([0], [0], color='#e8871e', lw=6, ls='-',  alpha=0.45, label='Mismatch'),
        ]
        axes[0].legend(handles=legend_lines, loc='upper right', framealpha=0.85, fontsize=9)

        # ─── Panel 2: GT field 填充 ───
        im1 = axes[1].imshow(Zi_gt, cmap='RdYlBu_r', vmin=0, vmax=1,
                             interpolation='bicubic', **imshow_args)
        self._contour_outlined(axes[1], Xi, Yi, Zi_gt, [0.5],
                               color='white', lw=1.8, ls='-')
        axes[1].set_title(f"GT {field_name}")
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
        cb1.set_label(field_name)

        # ─── Panel 3: Pred field 填充 ───
        im2 = axes[2].imshow(Zi_pred, cmap='RdYlBu_r', vmin=0, vmax=1,
                             interpolation='bicubic', **imshow_args)
        self._contour_outlined(axes[2], Xi, Yi, Zi_pred, [0.5],
                               color='white', lw=1.8, ls='-')
        # 叠加 GT 0.5 contour 用于对比 (黄色虚线, 带描边)
        self._contour_outlined(axes[2], Xi, Yi, Zi_gt, [0.5],
                               color='#FFD700', lw=1.5, ls='--')
        axes[2].set_title(f"Pred {field_name}")
        cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
        cb2.set_label(field_name)
        legend_pred = [
            Line2D([0], [0], color='white',   lw=1.8, ls='-',  label='Pred 0.5'),
            Line2D([0], [0], color='#FFD700', lw=1.5, ls='--', label='GT 0.5'),
        ]
        axes[2].legend(handles=legend_pred, loc='upper right', framealpha=0.85, fontsize=9)

        # ─── Panel 4: 界面附近误差热图 ───
        abs_err = np.abs(Zi_pred - Zi_gt)
        interface_region = (Zi_gt > 0.1) & (Zi_gt < 0.9)
        err_masked = np.where(interface_region, abs_err, np.nan)

        err_vmax = np.nanpercentile(err_masked, 99) if np.any(interface_region) else 0.1
        err_vmax = max(err_vmax, 0.01)

        im3 = axes[3].imshow(err_masked, cmap='inferno', vmin=0, vmax=err_vmax,
                             interpolation='bicubic', **imshow_args)
        self._contour_outlined(axes[3], Xi, Yi, Zi_gt, [0.5],
                               color='cyan', lw=1.8, ls='-')
        self._contour_outlined(axes[3], Xi, Yi, Zi_pred, [0.5],
                               color='lime', lw=1.8, ls='--')
        if np.any(disagree > 0.5):
            axes[3].contour(Xi, Yi, disagree, levels=[0.5], colors='yellow',
                            linewidths=1.0, linestyles=':', alpha=0.7)
        axes[3].set_title("Interface Error (|Pred-GT|)")
        cb3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.03, extend='max')
        cb3.set_label(f"|Pred - GT| ({field_name})")
        legend_err = [
            Line2D([0], [0], color='cyan', lw=1.8, ls='-',  label='GT 0.5'),
            Line2D([0], [0], color='lime', lw=1.8, ls='--', label='Pred 0.5'),
        ]
        axes[3].legend(handles=legend_err, loc='upper right', framealpha=0.85, fontsize=9)

        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(style='sci', scilimits=(-1, 1), axis='both')
            ax.set_aspect('equal', adjustable='box')

        fig.suptitle(f"{field_name} -- Step {time_step}", fontsize=14, fontweight='bold')

        # ─── 底部 Metrics 文本: 当前步 (上, 大字) + 全局均值 (下, 小字) ───
        if has_metrics:
            ax_text = fig.add_subplot(gs[1, :])
            ax_text.axis('off')
            # 当前步指标 — 较大字号、加粗
            ax_text.text(0.01, 0.95, metrics_text_step,
                         transform=ax_text.transAxes,
                         fontsize=12, fontfamily='monospace', fontweight='bold',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f0fe',
                                   edgecolor='#4a90d9', alpha=0.92))
            # 全局均值指标 — 稍小字号
            ax_text.text(0.01, 0.38, metrics_text_mean,
                         transform=ax_text.transAxes,
                         fontsize=11, fontfamily='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                                   edgecolor='#cccccc', alpha=0.90))

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

    # ──────────────────────────────
    #  GIF 生成
    # ──────────────────────────────

    def generate_gif(self, result, field_name=None, field_idx=None,
                     axis="z", slice_pos=None, gif_path="result/air_rollout.gif", res=280,
                     metrics=None):
        if field_name is None:
            field_name, field_idx = self.vof_fields[0]
        elif field_idx is None:
            field_idx = self.fields.index(field_name)

        print(f"[GIF] Generating interface animation for {field_name}...")
        frames = []
        horizon = result["pred"].shape[0]

        for t in tqdm(range(horizon), desc="Rendering"):
            img = self.plot_interface(
                result, time_step=t, field_name=field_name, field_idx=field_idx,
                axis=axis, slice_pos=slice_pos,
                res=res, return_array=True, metrics=metrics
            )
            if img is not None:
                frames.append(img)

        if frames:
            imageio.mimsave(gif_path, frames, fps=8, loop=0)
            print(f"[GIF] Saved to {gif_path}")
        else:
            print("[GIF] No frames generated.")

    # ──────────────────────────────
    #  汇总多样本指标
    # ──────────────────────────────

    def print_summary(self, all_metrics, out_dir=None):
        """汇总打印多个样本的平均指标，并可选同步写入 out_dir/metrics_summary.txt"""
        n = len(all_metrics)
        lines = []
        lines.append(f"\n{'#'*60}")
        lines.append(f"  SUMMARY over {n} samples")
        lines.append(f"{'#'*60}")

        mse_vals = [m["MSE_normalized"] for m in all_metrics]
        lines.append(f"  {'MSE_normalized':20s}:  mean={np.mean(mse_vals):.4e}  std={np.std(mse_vals):.4e}")

        field_names = list(all_metrics[0]["per_field"].keys())
        metric_keys = ["relative_L2", "RMSE", "mean_IoU", "mean_Dice", "mean_band_MAE"]
        for fname in field_names:
            lines.append(f"\n  --- {fname} ---")
            for k in metric_keys:
                vals = [m["per_field"][fname][k] for m in all_metrics]
                lines.append(f"    {k:20s}:  mean={np.mean(vals):.4e}  std={np.std(vals):.4e}  "
                             f"min={np.min(vals):.4e}  max={np.max(vals):.4e}")
        lines.append(f"{'#'*60}")

        text = "\n".join(lines)
        print(text)
        if out_dir is not None:
            txt_path = os.path.join(out_dir, "metrics_summary.txt")
            with open(txt_path, "w") as f:
                f.write(text + "\n")
            print(f"[Saved] {txt_path}")


# ════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════

if __name__ == "__main__":
    MODE = "test"
    # CONFIG_PATH = "config/config_alpha_air/easypool_air_3-7_enhanced.json"
    # CONFIG_PATH 也可以是 list，依次处理多个配置：
    CONFIG_PATH = [
        "config/keyhole/GTO_keyhole_stronger.json",
        "config/keyhole/GTO_attnres_keyhole_stronger.json",
        "config/keyhole/GTO_attnres_3_keyhole_stronger.json",
    ]

    SLICE_AXIS = "z"
    SLICE_POS = None
    NUM_SAMPLES = 3  # 每个 config 随机推理的样本数
    sample_idxs = [207, 229]

    cfg_list = CONFIG_PATH if isinstance(CONFIG_PATH, list) else [CONFIG_PATH]

    for cfg_path in cfg_list:
        try:
            predictor = AirFieldPredictor(cfg_path, MODE)
            OUT_DIR = f"result_keyhole/inference_standard/inference_air/{predictor.args.name}/{MODE}"
            os.makedirs(OUT_DIR, exist_ok=True)
        except Exception as e:
            print(f"Init failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        dataset_length = len(predictor.dataset)
        print(f"Dataset size: {dataset_length}")

        # sample_idxs = random.sample(range(dataset_length), min(NUM_SAMPLES, dataset_length))
        all_metrics = []

        for sample_idx in sample_idxs:
            print(f"\n>>> Inference on sample {sample_idx} ...")

            # 1. 推理 + 指标
            result, metrics = predictor.predict_and_evaluate(sample_idx)
            predictor.print_metrics(sample_idx, metrics, out_dir=OUT_DIR)
            all_metrics.append(metrics)

            # 2. 对每个 VOF 场分别生成 GIF
            for field_name, field_idx in predictor.vof_fields:
                safe_name = field_name.replace(".", "_")
                gif_path = os.path.join(OUT_DIR, f"interface_s{sample_idx}_{safe_name}.gif")
                predictor.generate_gif(
                    result,
                    field_name=field_name,
                    field_idx=field_idx,
                    axis=SLICE_AXIS,
                    slice_pos=SLICE_POS,
                    gif_path=gif_path,
                    metrics=metrics
                )

        # 3. 汇总
        predictor.print_summary(all_metrics, out_dir=OUT_DIR)
