# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import imageio
from tqdm import tqdm
from pathlib import Path
import h5py

# 引入项目模块
from src.dataset import AeroGtoDataset
from src.dataset_2d import AeroGtoDataset2D
from src.utils import load_json_config, set_seed


def _build_model(model_cfg, cond_dim, default_dt, device):
    """根据 config 中的 model.name 动态构建对应模型（兼容所有变体）"""
    model_name = model_cfg.get("name", "PhysGTO")

    if model_name == "PhysGTO":
        from src.physgto import Model
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

    if model_name in ("gto_attnres_multi", "gto_attnres_multi_v2", "gto_res_attnres"):
        kwargs["n_fields"] = model_cfg.get("n_fields", model_cfg.get("in_dim", 2))
        kwargs["cross_attn_heads"] = model_cfg.get("cross_attn_heads", 4)

    if model_name in ("gto_attnres_multi_v2", "gto_res_attnres"):
        kwargs["attn_res_mode"] = model_cfg.get("attn_res_mode", "block_inter")

    return Model(**kwargs).to(device)

class AeroGtoPredictor:
    def __init__(self, config_path, mode="test", model_path=None, device_str="cuda"):
        self.args = load_json_config(config_path)
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")

        data_cfg = self.args.data

        # 1. 初始化数据集
        print("[Init] Loading Test Dataset...")
        train_dataset = AeroGtoDataset(
            file_list=data_cfg["train_list"],
            mode="train",
            fields=data_cfg.get("fields", ["T"]),
            input_steps=data_cfg.get("input_steps", 1),
            horizon=data_cfg.get("horizon_test", 1),
            time_stride=data_cfg.get("time_stride", 1),
            spatial_stride=data_cfg.get("spatial_stride", 1),
            normalize=data_cfg.get("normalize", True),
            samples_per_file=data_cfg.get("samples_per_file", 32),
            norm_cache=data_cfg.get("norm_cache"),
        )

        if mode == "test":
            self.dataset = AeroGtoDataset(
                file_list=self.args.data["test_list"],
                mode="test",
                fields=self.args.data.get("fields", ["T"]),
                input_steps=self.args.data.get("input_steps", 1),
                horizon=self.args.data.get("horizon_test", 10),
                time_stride=self.args.data.get("time_stride", 1),
                spatial_stride=self.args.data.get("spatial_stride", 1),
                normalize=self.args.data.get("normalize", True),
                norm_cache=self.args.data.get("norm_cache"),
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
        model_cfg = self.args.model

        self.model = _build_model(model_cfg, cond_dim, default_dt, self.device)

        # 3. 加载权重（支持 EMA）
        if model_path is None:
            save_root = Path(self.args.save_path)
            model_path = save_root / "nn" / f"{self.args.name}_best.pt"

        print(f"[Init] Loading weights from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weight file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

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
        即使模型没有训练预测 alpha.air，我们也可以读取 GT 来画出真实的熔池轮廓
        """
        # 1. 找到对应的文件和时间步
        file_id, start_idx = self.dataset.sample_keys[sample_idx]
        path = self.dataset.file_paths[file_id]
        meta = self.dataset.meta_cache[path]
        
        # 2. 确定时间索引
        if start_idx is None: start_idx = 1
        
        # 计算对应的时间索引序列
        time_indices = start_idx + np.arange(1, self.dataset.horizon + 1) * self.dataset.time_stride
        indices = meta["indices"]
        
        try:
            with h5py.File(path, 'r') as f:
                if f"state/{interface_name}" not in f:
                    return None
                
                # 读取数据
                data = f[f"state/{interface_name}"][time_indices] # [Horizon, All_Points, 1]
                data = data[:, indices, 0] # 下采样 [Horizon, N]
                return data
        except Exception as e:
            print(f"[Warn] Failed to load GT interface '{interface_name}': {e}")
            return None

    def predict_rollout(self, sample_idx, interface_field="alpha.air"):
        """
        执行自回归预测，并准备绘图所需的所有数据
        """
        sample = self.dataset[sample_idx]
        
        # 增加 Batch 维度并移至 GPU
        state_seq = sample["state"].unsqueeze(0).to(self.device)
        node_pos = sample["node_pos"].unsqueeze(0).to(self.device)
        edges = sample["edges"].unsqueeze(0).to(self.device)
        time_seq = sample["time_seq"].unsqueeze(0).to(self.device) 
        conditions = sample["conditions"].unsqueeze(0).to(self.device).float()

        state_0 = state_seq[:, 0] 
        gt_seq = state_seq[:, 1:]

        print(f"[Predict] Running autoregressive inference...")
        with torch.no_grad():
            pred_seq = self.model.autoregressive(
                state_0, node_pos, edges, time_seq, conditions
            )
            
            pred_real = self.normalizer.denormalize(pred_seq)
            gt_real = self.normalizer.denormalize(gt_seq)

        # --- 边界数据获取逻辑 ---
        # 1. GT 边界：直接从 H5 文件读取 (最稳妥，不管模型训没训)
        gt_interface = self._load_gt_interface(sample_idx, interface_field)
        
        # 2. Pred 边界：只有当模型训练了 interface_field 时才有
        pred_interface = None
        if interface_field in self.fields:
            idx = self.fields.index(interface_field)
            pred_interface = pred_real[0, :, :, idx].cpu().numpy() # [Horizon, N]
        else:
            print(f"[Info] Model does not predict '{interface_field}', so Pred boundary will be missing.")
        
        return {
            "pred": pred_real.cpu().numpy()[0], # [T, N, C]
            "gt": gt_real.cpu().numpy()[0],     # [T, N, C]
            "coords": node_pos.cpu().numpy()[0], # [N, 3]
            "gt_interface": gt_interface,       # [T, N] or None
            "pred_interface": pred_interface    # [T, N] or None
        }

    def plot_slice(self, 
                   result_dict, 
                   time_step, 
                   field_name="T", 
                   axis="z", 
                   slice_pos=None, 
                   res=300, # 分辨率，越高越细腻
                   vmin=None, vmax=None, 
                   return_array=False,
                   save_path=None,
                   interface=True):
        
        # 1. 准备主物理场数据
        try:
            f_idx = self.fields.index(field_name)
        except ValueError:
            print(f"Field {field_name} not found.")
            return
        
        if field_name == "alpha.air" or field_name == "F_sum" or field_name == "gamma_liquid":
            vmax, vmin = 1, 0

        pred_data = result_dict["pred"][time_step, :, f_idx]
        gt_data = result_dict["gt"][time_step, :, f_idx]
        coords = result_dict["coords"]
        
        # 准备界面场数据
        gt_int = None
        pred_int = None
        if result_dict.get("gt_interface") is not None:
            gt_int = result_dict["gt_interface"][time_step]
        if result_dict.get("pred_interface") is not None:
            pred_int = result_dict["pred_interface"][time_step]

        # 2. 确定切片掩码
        if slice_pos is None:
            axis_idx = {'x':0, 'y':1, 'z':2}[axis]
            slice_pos = (coords[:, axis_idx].max() + coords[:, axis_idx].min()) / 2

        # 自动调整 thickness (简单的自适应)
        axis_id = {'x':0, 'y':1, 'z':2}[axis]
        # thickness = 1e-8 # 原来的太薄可能切不到点，建议稍大一点或者保留你原本测试好的值
        thickness = 1e-6 

        mask = np.abs(coords[:, axis_id] - slice_pos) < thickness
        
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

        if np.sum(mask) < 10:
            print(f"[Error] Too few points in slice {axis}={slice_pos:.3e}.")
            return

        # 3. 插值 (Interpolation)
        xi = np.linspace(pts_x.min(), pts_x.max(), res)
        yi = np.linspace(pts_y.min(), pts_y.max(), res)
        Xi, Yi = np.meshgrid(xi, yi)

        def interp(vals):
            if vals is None: return None
            # 'linear' 插值比较平滑
            return griddata((pts_x, pts_y), vals[mask], (Xi, Yi), method='linear')

        Zi_pred = interp(pred_data)
        Zi_gt = interp(gt_data)
        
        # --- 误差计算 (核心修改) ---
        # 1. 绝对误差
        Zi_err = np.abs(Zi_pred - Zi_gt)
        
        # 2. 相对误差 (带背景掩码)
        Zi_rel_err = np.zeros_like(Zi_err)
        
        # 动态计算阈值：取数据范围的 1%
        # 对于 VOF (0~1)，阈值就是 0.01；对于温度 (300~3000)，阈值就是 27
        data_range = np.nanmax(Zi_gt) - np.nanmin(Zi_gt)
        threshold = 0.01 * data_range if data_range > 0 else 1e-6
        
        # 掩码：只在 GT 值显著的地方计算相对误差
        bg_mask = np.abs(Zi_gt) > threshold
        
        # 只有 mask 为 True 的地方才做除法，其他地方保持为 0 (蓝色背景)
        if np.any(bg_mask):
             Zi_rel_err[bg_mask] = Zi_err[bg_mask] / (np.abs(Zi_gt[bg_mask]) + 1e-8)
        
        # 插值界面场
        if interface:
            Zi_gt_int = interp(gt_int)
            Zi_pred_int = interp(pred_int)
        else:
            Zi_gt_int, Zi_pred_int = None, None

        # 4. 绘图 (修改为 1x4 布局)
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        
        if field_name == "alpha.air": vmin, vmax = 0, 1
        if vmin is None: vmin = min(np.nanmin(Zi_gt), np.nanmin(Zi_pred))
        if vmax is None: vmax = max(np.nanmax(Zi_gt), np.nanmax(Zi_pred))

        cmap = 'inferno' if 'T' in field_name else 'viridis'

        # --- Subplot 1: Ground Truth ---
        im0 = axes[0].imshow(Zi_gt, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Ground Truth ({field_name})")
        plt.colorbar(im0, ax=axes[0])
        if Zi_gt_int is not None:
            axes[0].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
        
        # --- Subplot 2: Prediction ---
        im1 = axes[1].imshow(Zi_pred, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Prediction ({field_name})")
        plt.colorbar(im1, ax=axes[1])
        if Zi_gt_int is not None:
            axes[1].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
        
        # --- Subplot 3: Absolute Error ---
        im2 = axes[2].imshow(Zi_err, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', cmap='coolwarm')
        axes[2].set_title("Absolute Error")
        plt.colorbar(im2, ax=axes[2])

        # --- [新增] Subplot 4: Relative Error ---
        # vmin=0 确保背景是中性色/冷色，vmax=1.0 限制最大显示 100% 误差，避免极大值破坏对比度
        im3 = axes[3].imshow(Zi_rel_err, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', cmap='coolwarm')
        axes[3].set_title("Relative Error (>1% region)")
        plt.colorbar(im3, ax=axes[3])

        # 准备图例对象
        lines = []
        labels = []
        if Zi_gt_int is not None:
            lines.append(plt.Line2D([0], [0], color='white', linestyle='--', linewidth=1.5))
            labels.append('GT Interface')
        if Zi_pred_int is not None:
            lines.append(plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5))
            labels.append('Pred Interface')

        # 在 Abs Error (axes[2]) 和 Rel Error (axes[3]) 上都画边界和图例
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

        plt.tight_layout()

        # 5. 输出
        if return_array:
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            img = imageio.v2.imread(buf)
            buf.close()
            return img
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Saved] {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def generate_gif(self, result_dict, field_name="T", axis="z", slice_pos=None, git_path="result/inference_results", interface=True):
        print(f"[GIF] Generating animation for field '{field_name}'...")
        f_idx = self.fields.index(field_name)
        all_vals = np.concatenate([result_dict["pred"][..., f_idx], result_dict["gt"][..., f_idx]])
        vmin, vmax = 0, 4000
        print(f"[GIF] Fixed scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

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
                interface=interface
            )
            frames.append(img)
        
        imageio.mimsave(git_path, frames, fps=10, loop=0)
        print(f"[GIF] Saved to {git_path}")

if __name__ == "__main__":
    MODE = "test"
    NAME= "aerogto_large_T_air_normlize_with_new_loss"
    # === 配置区域 ===
    # CONFIG_PATH = f"config/{NAME if NAME is not None and NAME != '' else 'aerogto_base'}.json" 
    CONFIG_PATH = f"config/aerogto_large_patch_version.json"
    SAMPLE_IDX = 50    
    FIELD_TO_PLOT = "T"     
    SLICE_AXIS = "z"        # 'x', 'y', 'z'
    SLICE_POS = None        
    INTERFACE_FIELD = "alpha.air" # 定义用来画分界线的物理场

    try:
        predictor = AeroGtoPredictor(CONFIG_PATH, MODE)
        OUT_DIR = f"result/inference_results/{predictor.args.name}/{FIELD_TO_PLOT}/{MODE}"
        os.makedirs(OUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 1. 执行推理 (自动获取主物理场和边界场)
    results = predictor.predict_rollout(sample_idx=SAMPLE_IDX, interface_field=INTERFACE_FIELD)
    
    # 2. 生成单帧图片 (例如第 5 步)
    # for step in range(10):
    #     # if step < results["pred"].shape[0]:
    #     save_p = os.path.join(OUT_DIR, f"snapshot_sample{SAMPLE_IDX}_step{step}_{FIELD_TO_PLOT}_{SLICE_AXIS}.png")
    #     predictor.plot_slice(results, time_step=step, field_name=FIELD_TO_PLOT, axis=SLICE_AXIS, slice_pos=SLICE_POS,
    #                             vmin=0, vmax=4000, save_path=save_p)

    # 3. 生成 GIF
    gif_path = os.path.join(OUT_DIR, f"rollout_sample{SAMPLE_IDX}_{FIELD_TO_PLOT}.gif")
    predictor.generate_gif(
        results, 
        field_name=FIELD_TO_PLOT, 
        axis=SLICE_AXIS, 
        slice_pos=SLICE_POS, 
        git_path=gif_path,
        interface=False
    )