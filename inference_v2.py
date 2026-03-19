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

# 引入项目模块
from src.physgto_res import Model
from src.dataset import AeroGtoDataset
from src.utils import load_json_config, set_seed

class AeroGtoPredictor:
    def __init__(self, config_path, mode="test", model_path=None, device_str="cuda"):
        self.args = load_json_config(config_path)
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")

        data_cfg = self.args.data
        model_cfg = self.args.model
        self.plot_cfg = getattr(self.args, "plot", {}) or {}
        self.plot_field_limits = self.plot_cfg.get("field_limits", {}) or {}

        self.fields = data_cfg.get("fields", ["T"])
        space_dim = model_cfg.get("space_size", 3)
        
        print("[Init] Loading Train Dataset (for Normalizer)...")
        # 即使是 inference，通常也需要 TrainSet 的统计数据来做 Normalizer
        train_dataset = AeroGtoDataset(
            config=data_cfg,
            mode="train",
        )

        if mode == "test":
            print("[Init] Loading Test Dataset...")
            self.dataset = AeroGtoDataset(
                config=data_cfg,
                mode="test",
                mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None,
            )
            self.dataset.normalizer = train_dataset.normalizer
        elif mode == "train":
            self.dataset = train_dataset

        self.fields = self.dataset.fields
        print(f"[Init] Fields to predict: {self.fields}")

        # 2. 构建模型
        print("[Init] Building Model...")
        cond_dim = self.args.model.get("cond_dim") or self.dataset.cond_dim
        default_dt = self.args.model.get("dt", self.dataset.dt)
        
        self.model = Model(
            space_size=self.args.model.get("space_size", 3),
            pos_enc_dim=self.args.model.get("pos_enc_dim", 5),
            cond_dim=cond_dim,
            N_block=self.args.model.get("N_block", 4),
            in_dim=self.args.model.get("in_dim", 4),
            out_dim=self.args.model.get("out_dim", 4),
            enc_dim=self.args.model.get("enc_dim", 128),
            n_head=self.args.model.get("n_head", 4),
            n_token=self.args.model.get("n_token", 64),
            dt=self.args.model.get("dt", default_dt),
        ).to(self.device)

        # 3. 加载权重
        if model_path is None:
            save_root = Path(self.args.save_path)
            model_path = save_root / "nn" / f"{self.args.name}_best.pt"
        
        print(f"[Init] Loading weights from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weight file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        self.normalizer = self.dataset.normalizer
        self.normalizer.to(self.device)

    def _load_gt_interface(self, sample_idx, interface_name="alpha.air"):
        """辅助函数：额外从H5文件中读取真实的界面场 (GT)，用于画图"""
        file_id, start_idx = self.dataset.sample_keys[sample_idx]
        path = self.dataset.file_paths[file_id]
        meta = self.dataset.meta_cache[path]
        
        if start_idx is None: start_idx = 1
        
        time_indices = start_idx + np.arange(1, self.dataset.horizon + 1) * self.dataset.time_stride
        indices = meta["indices"]
        
        try:
            with h5py.File(path, 'r') as f:
                if f"state/{interface_name}" not in f:
                    return None
                data = f[f"state/{interface_name}"][time_indices]
                data = data[:, indices, 0] 
                return data
        except Exception as e:
            print(f"[Warn] Failed to load GT interface '{interface_name}': {e}")
            return None

    def predict_rollout(self, sample_idx, interface_field="alpha.air"):
        """执行自回归预测，并准备绘图所需的所有数据"""
        sample = self.dataset[sample_idx]
        
        state_seq = sample["state"].unsqueeze(0).to(self.device)
        node_pos = sample["node_pos"].unsqueeze(0).to(self.device)
        edges = sample["edges"].unsqueeze(0).to(self.device)
        time_seq = sample["time_seq"].unsqueeze(0).to(self.device) 
        conditions = sample["conditions"].unsqueeze(0).to(self.device).float()
        
        dt = sample["dt"]
        state_0 = state_seq[:, 0] 
        gt_seq = state_seq[:, 1:]

        print(f"[Predict] Running autoregressive inference...")
        with torch.no_grad():
            pred_seq = self.model.autoregressive(
                state_0, node_pos, edges, time_seq, conditions, dt
            )
            pred_real = self.normalizer.denormalize(pred_seq)
            gt_real = self.normalizer.denormalize(gt_seq)

        file_id, _ = self.dataset.sample_keys[sample_idx]
        path = self.dataset.file_paths[file_id]
        meta = self.dataset.meta_cache[path]
        
        raw_coords = meta["node_pos"]
        if isinstance(raw_coords, torch.Tensor):
            raw_coords = raw_coords.cpu().numpy()

        gt_interface = self._load_gt_interface(sample_idx, interface_field)
        
        pred_interface = None
        if interface_field in self.fields:
            idx = self.fields.index(interface_field)
            pred_interface = pred_real[0, :, :, idx].cpu().numpy() 
        else:
            print(f"[Info] Model does not predict '{interface_field}', so Pred boundary will be missing.")
        
        return {
            "pred": pred_real.cpu().numpy()[0], 
            "gt": gt_real.cpu().numpy()[0],     
            "coords": raw_coords,               
            "gt_interface": gt_interface,       
            "pred_interface": pred_interface    
        }
    
    def _is_phase_field(self, field_name):
        return any(k in field_name for k in ["alpha", "gamma", "frac"])

    def _is_velocity_field(self, field_name):
        return field_name in ["Ux", "Uy", "Uz", "U"]

    def _resolve_field_limits(self, result_dict, field_name, field_idx, interface=True):
        cfg_limits = self.plot_field_limits.get(field_name)
        if cfg_limits is None and self._is_velocity_field(field_name):
            cfg_limits = self.plot_cfg.get("velocity_limits")

        if isinstance(cfg_limits, (list, tuple)) and len(cfg_limits) == 2:
            return float(cfg_limits[0]), float(cfg_limits[1]), True

        if field_name == "T":
            return 300.0, 3500.0, False
        if field_name in ["alpha.air", "alpha.titanium", "gamma_liquid"]:
            return 0.0, 1.0, True

        if self._is_velocity_field(field_name):
            if interface and result_dict.get("gt_interface") is not None:
                gas_mask_all = result_dict["gt_interface"] > 0.5
                valid_pred = np.where(gas_mask_all, np.nan, result_dict["pred"][..., field_idx])
                valid_gt = np.where(gas_mask_all, np.nan, result_dict["gt"][..., field_idx])
            else:
                valid_pred = result_dict["pred"][..., field_idx]
                valid_gt = result_dict["gt"][..., field_idx]

            combined = np.concatenate([valid_pred.reshape(-1), valid_gt.reshape(-1)])
            active_combined = combined[np.abs(combined) > 1e-4]

            if len(active_combined) < 10 or np.all(np.isnan(active_combined)):
                return -1.0, 1.0, False

            v_min_raw = np.nanmin(active_combined)
            v_max_raw = np.nanmax(active_combined)
            v_abs_max = max(abs(v_min_raw), abs(v_max_raw), 1e-3)
            return -v_abs_max, v_abs_max, False

        all_pred = result_dict["pred"][..., field_idx]
        all_gt = result_dict["gt"][..., field_idx]
        combined = np.concatenate([all_pred.reshape(-1), all_gt.reshape(-1)])
        return float(np.nanmin(combined)), float(np.nanmax(combined)), False

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

    def _interp_grid(self, pts_x, pts_y, vals, Xi, Yi, method="cubic"):
        if vals is None:
            return None
            
        # 1. 组合成 2D 坐标点对
        points_2d = np.column_stack([pts_x, pts_y])
        values = np.asarray(vals)

        # 2. 【终极杀手锏】：剔除投影后完全重合的重复点
        points_unique, unique_indices = np.unique(points_2d, axis=0, return_index=True)
        values_unique = values[unique_indices]

        # 3. 加上微小扰动防止共线退化
        jitter = np.random.normal(0, 1e-10, size=points_unique.shape)
        points_jittered = points_unique + jitter

        interp_try = [method]
        if method != "linear":
            interp_try.append("linear")
        interp_try.append("nearest")

        Z = None
        for m in interp_try:
            try:
                Z = griddata(points_jittered, values_unique, (Xi, Yi), method=m)
            except Exception:
                Z = None
            if Z is not None and not np.all(np.isnan(Z)):
                break

        if Z is None:
            return None

        if np.isnan(Z).any():
            try:
                Z_nearest = griddata(points_jittered, values_unique, (Xi, Yi), method="nearest")
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
                   res=320, 
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

        is_velocity = self._is_velocity_field(field_name)
        
        if vmin is None or vmax is None:
            vmin, vmax, clip_to_range = self._resolve_field_limits(result_dict, field_name, f_idx, interface=interface)
        else:
            clip_to_range = True

        pred_data = result_dict["pred"][time_step, :, f_idx].copy()
        gt_data = result_dict["gt"][time_step, :, f_idx].copy()
        coords = result_dict["coords"] 
        
        gt_int = None
        pred_int = None
        if result_dict.get("gt_interface") is not None:
            gt_int = result_dict["gt_interface"][time_step]
        if result_dict.get("pred_interface") is not None:
            pred_int = result_dict["pred_interface"][time_step]

        # 速度场气态镂空 (NaN) 与 固态置零 (0.0)
        if is_velocity:
            # 1. 气态区域镂空
            if gt_int is not None:
                gas_mask = gt_int > 0.5
                pred_data[gas_mask] = np.nan
                gt_data[gas_mask] = np.nan
            
            # 2. 固态区域强行置 0 (清除基底噪声)
            if "gamma_liquid" in self.fields:
                gamma_idx = self.fields.index("gamma_liquid")
                gt_gamma = result_dict["gt"][time_step, :, gamma_idx]
                solid_mask = gt_gamma < 0.1
                
                # 避开已经被设为 NaN 的气相区域
                solid_valid_mask_pred = solid_mask & ~np.isnan(pred_data)
                pred_data[solid_valid_mask_pred] = 0.0
                
                solid_valid_mask_gt = solid_mask & ~np.isnan(gt_data)
                gt_data[solid_valid_mask_gt] = 0.0

        mask, slice_pos = self._build_slice_mask(coords, axis, slice_pos=slice_pos, min_points=32)
        
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

        xi = np.linspace(pts_x.min(), pts_x.max(), res)
        yi = np.linspace(pts_y.min(), pts_y.max(), res)
        Xi, Yi = np.meshgrid(xi, yi)

        # 恢复相场用 linear，速度/温度用 cubic 的平滑逻辑
        is_phase = self._is_phase_field(field_name)
        interp_method = "linear" if is_phase else "cubic"
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

        Zi_pred_vis = Zi_pred
        Zi_gt_vis = Zi_gt
        colorbar_extend = 'neither'
        if clip_to_range:
            Zi_pred_vis = np.clip(Zi_pred, vmin, vmax)
            Zi_gt_vis = np.clip(Zi_gt, vmin, vmax)
            colorbar_extend = 'both'
        
        Zi_err = np.abs(Zi_pred - Zi_gt)
        denom_floor = max(1e-8, 1e-3 * np.nanmax(np.abs(Zi_gt)))
        Zi_rel_err = Zi_err / np.maximum(np.abs(Zi_gt), denom_floor)
        Zi_rel_err = np.clip(Zi_rel_err, 0, rel_err_cap)

        Zi_err_vis = self._smooth_for_plot(Zi_err, sigma=min(max(sigma, 0.3), 1.0))
        Zi_rel_err_vis = self._smooth_for_plot(Zi_rel_err, sigma=min(max(sigma, 0.3), 1.0))
        
        if interface:
            Zi_gt_int = self._interp_grid(pts_x, pts_y, gt_int[mask] if gt_int is not None else None, Xi, Yi, method="linear")
            Zi_pred_int = self._interp_grid(pts_x, pts_y, pred_int[mask] if pred_int is not None else None, Xi, Yi, method="linear")
        else:
            Zi_gt_int, Zi_pred_int = None, None

        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
        fig, axes = plt.subplots(1, 4, figsize=(21, 5), constrained_layout=True)
        
        if vmin is None: vmin = np.nanmin(Zi_gt)
        if vmax is None: vmax = np.nanmax(Zi_gt)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9

        if is_velocity:
            cmap = "RdBu_r"
        elif "T" in field_name:
            cmap = "inferno"
        else:
            cmap = "viridis"
            
        imshow_args = dict(extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', interpolation='bicubic', aspect='equal')

        im0 = axes[0].imshow(Zi_gt_vis, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_args)
        axes[0].set_title(f"Ground Truth ({field_name})")
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, extend=colorbar_extend)
        cb0.set_label(field_name)
        
        im1 = axes[1].imshow(Zi_pred_vis, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_args)
        axes[1].set_title(f"Prediction ({field_name})")
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03, extend=colorbar_extend)
        cb1.set_label(field_name)
        
        err_vmax = np.nanpercentile(Zi_err_vis, 99.0) if np.any(np.isfinite(Zi_err_vis)) else 1.0
        err_vmax = max(err_vmax, 1e-12)
        im2 = axes[2].imshow(Zi_err_vis, cmap='magma', vmin=0, vmax=err_vmax, **imshow_args)
        axes[2].set_title("Absolute Error")
        cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03, extend='max')
        cb2.set_label("|Pred - GT|")

        rel_vmax = min(rel_err_cap, max(0.1, np.nanpercentile(Zi_rel_err_vis, 99.0)))
        im3 = axes[3].imshow(Zi_rel_err_vis, cmap='magma', vmin=0, vmax=rel_vmax, **imshow_args)
        axes[3].set_title(f"Relative Error (clip={rel_err_cap:.2f})")
        cb3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.03, extend='max')
        cb3.set_label("|Pred-GT| / max(|GT|, eps)")

        lines, labels = [], []
        if Zi_gt_int is not None:
            axes[0].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white' if "T" in field_name else 'black', linestyles='--', linewidths=1.5)
            axes[1].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white' if "T" in field_name else 'black', linestyles='--', linewidths=1.5)
            lines.append(plt.Line2D([0], [0], color='white' if "T" in field_name else 'black', linestyle='--', linewidth=1.5))
            labels.append('GT Interface')
            
        if Zi_pred_int is not None:
             lines.append(plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5))
             labels.append('Pred Interface')

        for ax in [axes[2], axes[3]]:
            if Zi_gt_int is not None:
                ax.contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white' if "T" in field_name else 'black', linestyles='--', linewidths=1.5)
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
            plt.close(fig)
        else:
            plt.show()

    def generate_gif(self, result_dict, field_name="T", axis="z", slice_pos=None, git_path="result/inference_results", interface=True):
        print(f"[GIF] Generating animation for field '{field_name}'...")
        
        f_idx = self.fields.index(field_name)
        vmin, vmax, clip_to_range = self._resolve_field_limits(result_dict, field_name, f_idx, interface=interface)
        if clip_to_range:
            print(f"[GIF] Using clipped limits for {field_name}: {vmin:.4f} ~ {vmax:.4f}")
        elif self._is_velocity_field(field_name):
            print(f"[GIF] Velocity auto-detected symmetric limits: {vmin:.4f} ~ {vmax:.4f}")
        else:
            print(f"[GIF] Auto-detected global limits: {vmin:.4f} ~ {vmax:.4f}")

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
    CONFIG_PATH = f"config/EasyPool_2D_solid_mask_0316/aerogto_easypool_soild_mask.json" 
    
    FIELD_TO_PLOT = None   
    SLICE_AXIS = "z"        # 'x', 'y', 'z'
    SLICE_POS = 5e-4        
    INTERFACE_FIELD = "alpha.air" 

    try:
        predictor = AeroGtoPredictor(CONFIG_PATH, MODE)
        if FIELD_TO_PLOT is None:
            OUT_DIR = f"result/inference_results/{predictor.args.name}/{MODE}/batch"
        else:
            OUT_DIR = f"result/inference_results/{predictor.args.name}/{MODE}/{FIELD_TO_PLOT}"
        os.makedirs(OUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(len(predictor.dataset))
    SAMPLE_IDX = 27

    results = predictor.predict_rollout(sample_idx=SAMPLE_IDX, interface_field=INTERFACE_FIELD)
    
    if FIELD_TO_PLOT is None:
        for field in predictor.fields:
            gif_path = os.path.join(OUT_DIR, f"rollout_sample{SAMPLE_IDX}_{field}.gif")
            predictor.generate_gif(
                results, 
                field_name=field, 
                axis=SLICE_AXIS, 
                slice_pos=SLICE_POS, 
                git_path=gif_path,
                interface=True
            )
    else:
        gif_path = os.path.join(OUT_DIR, f"rollout_sample{SAMPLE_IDX}_{FIELD_TO_PLOT}.gif")
        if os.path.exists(gif_path):
            exit()
        predictor.generate_gif(
            results, 
            field_name=FIELD_TO_PLOT, 
            axis=SLICE_AXIS, 
            slice_pos=SLICE_POS, 
            git_path=gif_path,
            interface=True
        )
