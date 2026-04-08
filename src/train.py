import math
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from pathlib import Path
from torch.amp import GradScaler, autocast # 引入 AMP 模块

# l2 误差计算需要反归一化数据
def _relative_l2(pred, target):
    """相对L2误差，返回 [batch] 张量。"""
    error = pred - target
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-6)
    norm_error_channel = torch.mean(norm_error, dim=-1)  # 平均所有通道
    norm_error_time = torch.mean(norm_error_channel, dim=-1)
    return norm_error_time


def _each_l2(pred, target):
    t_step = target.shape[1]
    losses_each_t = torch.zeros(t_step, device=pred.device)
    for t in range(t_step):
        error = pred[:, t] - target[:, t]
        norm_error = torch.norm(error, dim=-2) / (torch.norm(target[:, t], dim=-2) + 1e-6)
        norm_error_channel = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_channel, dim=0)
        losses_each_t[t] = norm_error_batch
    return losses_each_t

# rmse 同样需要反归一化数据
def _rmse(pred, target):
    diff = pred - target
    mse = torch.mean(diff**2, dim=[0, 1, 2])
    return torch.sqrt(mse)

# ---- Region-masked metric helpers ----

def _masked_relative_l2(pred, target, mask):
    """
    Relative L2 error over masked spatial points for a single channel.
    Args:
        pred, target: [B, T, N, 1] denormalized
        mask: [B, T, N, 1] bool
    Returns:
        float scalar, or float('nan') if mask has zero True entries.
    """
    if mask.sum() == 0:
        return float('nan')
    error = (pred - target) * mask.float()
    target_masked = target * mask.float()
    error_norm = torch.norm(error, dim=-2)          # [B, T, 1]
    target_norm = torch.norm(target_masked, dim=-2)  # [B, T, 1]
    rel = error_norm / (target_norm + 1e-6)
    return rel.mean().item()


def _masked_rmse(pred, target, mask):
    """
    RMSE over masked spatial points for a single channel.
    Args:
        pred, target: [B, T, N, 1] denormalized
        mask: [B, T, N, 1] bool
    Returns:
        float scalar, or float('nan') if mask is empty.
    """
    count = mask.sum()
    if count == 0:
        return float('nan')
    diff = (pred - target) * mask.float()
    mse = (diff ** 2).sum() / count
    return torch.sqrt(mse).item()


def _masked_mse(pred, target, mask):
    """
    MSE over masked region (monitoring only, not for backprop).
    Args:
        pred, target: [B, T, N, C] (normalized)
        mask: [B, T, N, C] bool
    Returns:
        float scalar.
    """
    count = mask.sum()
    if count == 0:
        return 0.0
    diff = (pred - target) * mask.float()
    return ((diff ** 2).sum() / count).item()

def compute_spatial_gradient_3d(tensor_field, grid_shape):
    """
    计算 3D 空间梯度 (一阶有限差分)
    :param tensor_field: 形状为 [B, T, N, 1] 的展平张量
    :param grid_shape: (Nx, Ny, Nz) 三维网格尺寸
    :return: 沿 x, y, z 三个维度的梯度张量
    """
    B, T, N, C = tensor_field.shape

    Nx, Ny, Nz = int(grid_shape[0][0]), int(grid_shape[0][1]), int(grid_shape[0][2])
    
    # 确保网格尺寸和展平尺寸匹配
    assert Nx * Ny * Nz == N, f"网格尺寸 {grid_shape} 与节点数 {N} 不匹配！"
    
    # 1. 还原为 3D 网格形状: [B, Nx, Ny, Nz, C]
    grid_field = tensor_field.view(B, T, Nz, Ny, Nx, C)
    
    # 2. 沿三个空间维度计算一阶差分
    grad_x = grid_field[:, :, :, :, 1:, :] - grid_field[:, :, :, :, :-1, :]
    grad_y = grid_field[:, :, :, 1:, :, :] - grid_field[:, :, :, :-1, :, :]
    grad_z = grid_field[:, :, 1:, :, :, :] - grid_field[:, :, :-1, :, :, :]
    
    return grad_x, grad_y, grad_z


# ---- 场类型判定 ----

def _is_vof_field(field_name: str) -> bool:
    """判断是否为体积分数场（Volume-of-Fluid）。
    以 'alpha' 或 'gamma' 开头的场视为体积分数场。"""
    return field_name.startswith("alpha") or field_name.startswith("gamma")


# ---- 构建空间权重 mask ----

def _build_weight_mask(fld_gt_real, thresh, bw, fw):
    """
    根据真实值和阈值构建空间权重 mask。
    Args:
        fld_gt_real: [B, T, N, 1] 反归一化后的真实值
        thresh: float 或 [low, high] 列表
        bw: float, 背景区域权重
        fw: float, 聚焦区域权重
    Returns:
        weight_mask: [B, T, N, 1]，均值归一化后的权重
    """
    with torch.no_grad():
        if isinstance(thresh, list):
            condition = (fld_gt_real > thresh[0]) & (fld_gt_real < thresh[1])
            weight_mask = torch.where(condition, fw, bw)
        else:
            weight_mask = torch.where(fld_gt_real > thresh, fw, bw)
        weight_mask = weight_mask / torch.mean(weight_mask)
    return weight_mask


# ---- 单场 value loss 计算 ----

def _compute_weighted_value_loss(fld_pred, fld_gt, fld_gt_real, thresh, bw, fw, is_vof):
    """
    计算单个场的加权 value loss。
    Args:
        fld_pred, fld_gt: [B, T, N, 1] 归一化后的预测和标签
        fld_gt_real: [B, T, N, 1] 反归一化后的真实标签（用于构建 weight mask）
        thresh: float 或 [low, high]
        bw, fw: float, base/focus 权重
        is_vof: bool, 是否为体积分数场
    Returns:
        (loss_tensor, weight_mask):
            loss_tensor: [B, T, N, 1] element-wise 加权损失
            weight_mask: [B, T, N, 1] 空间权重（供梯度损失复用）
    """
    weight_mask = _build_weight_mask(fld_gt_real, thresh, bw, fw)

    if is_vof:
        # Weighted Huber Loss：对小误差保持 MSE 灵敏度，对大误差用线性避免梯度爆炸
        error = fld_pred - fld_gt
        abs_error = torch.abs(error)
        with torch.no_grad():
            # 自适应 delta：取当前 batch 误差中位数，clamp 防止退化
            delta = torch.clamp(torch.median(abs_error), min=1e-3)
        huber = torch.where(
            abs_error <= delta,
            0.5 * error ** 2,
            delta * (abs_error - 0.5 * delta)
        )
        loss_tensor = huber * weight_mask
    else:
        # 普通场：加权 MSE
        squared_error = (fld_pred - fld_gt) ** 2
        loss_tensor = squared_error * weight_mask

    return loss_tensor, weight_mask


# ---- 单场梯度损失计算 ----

def _adapt_mask_to_gradient(weight_mask, grid_shape, axis):
    """
    将空间 weight_mask 从 [B, T, N, 1] 适配到梯度差分后的维度。
    差分沿 axis 方向会少一个元素，取相邻元素的最大值作为梯度位置的权重。
    Args:
        weight_mask: [B, T, N, 1]
        grid_shape: numpy array, 形如 [[Nx, Ny, Nz]]
        axis: 'x' | 'y' | 'z'
    Returns:
        适配后的 weight_mask，shape 与对应方向的梯度张量一致
    """
    B, T, N, C = weight_mask.shape
    Nx, Ny, Nz = int(grid_shape[0][0]), int(grid_shape[0][1]), int(grid_shape[0][2])
    # 还原为 3D: [B, T, Nz, Ny, Nx, C]
    grid_mask = weight_mask.view(B, T, Nz, Ny, Nx, C)

    if axis == 'x':
        # 差分沿 Nx，取相邻 max
        return torch.max(grid_mask[:, :, :, :, 1:, :], grid_mask[:, :, :, :, :-1, :])
    elif axis == 'y':
        return torch.max(grid_mask[:, :, :, 1:, :, :], grid_mask[:, :, :, :-1, :, :])
    elif axis == 'z':
        return torch.max(grid_mask[:, :, 1:, :, :, :], grid_mask[:, :, :-1, :, :, :])
    else:
        raise ValueError(f"未知 axis: {axis}")


def _compute_gradient_loss(fld_pred, fld_gt, grid_shape, gw, is_vof, weight_cfg, weight_mask=None):
    """
    计算单个场的空间梯度损失。
    Args:
        fld_pred, fld_gt: [B, T, N, 1] 归一化值
        grid_shape: numpy array
        gw: float, 该场的梯度权重系数
        is_vof: bool
        weight_cfg: dict, 完整的 weight_loss 配置
        weight_mask: [B, T, N, 1] 可选，来自 value loss 的空间权重
    Returns:
        loss: scalar tensor
    """
    pred_gx, pred_gy, pred_gz = compute_spatial_gradient_3d(fld_pred, grid_shape)
    gt_gx, gt_gy, gt_gz = compute_spatial_gradient_3d(fld_gt, grid_shape)

    if is_vof:
        mode = weight_cfg.get("grad_weight_mode", "mask")

        if mode == "adaptive":
            # 用 GT 梯度幅值作为权重：梯度越大 → 界面越锋利 → 权重越高
            loss_g = 0
            for pred_g, gt_g in [(pred_gx, gt_gx), (pred_gy, gt_gy), (pred_gz, gt_gz)]:
                with torch.no_grad():
                    gt_mag = torch.abs(gt_g)
                    # 归一化：w = 1 + |∇gt| / mean(|∇gt|)，确保平坦区权重≥1
                    w = 1.0 + gt_mag / (torch.mean(gt_mag) + 1e-8)
                    w = w / torch.mean(w)  # 总体均值归一化，保持 loss 量级稳定
                loss_g = loss_g + torch.mean(w * (pred_g - gt_g) ** 2)

        else:  # mode == "mask"
            if weight_mask is not None:
                # 将 weight_mask 适配到差分后的维度
                wmx = _adapt_mask_to_gradient(weight_mask, grid_shape, 'x')
                wmy = _adapt_mask_to_gradient(weight_mask, grid_shape, 'y')
                wmz = _adapt_mask_to_gradient(weight_mask, grid_shape, 'z')

                loss_g = (torch.mean(wmx * (pred_gx - gt_gx) ** 2)
                          + torch.mean(wmy * (pred_gy - gt_gy) ** 2)
                          + torch.mean(wmz * (pred_gz - gt_gz) ** 2))
            else:
                # fallback: 无 weight_mask 时退化为普通 MSE
                loss_g = (F.mse_loss(pred_gx, gt_gx)
                          + F.mse_loss(pred_gy, gt_gy)
                          + F.mse_loss(pred_gz, gt_gz))
    else:
        # 普通场：纯 MSE 梯度损失
        loss_g = (F.mse_loss(pred_gx, gt_gx)
                  + F.mse_loss(pred_gy, gt_gy)
                  + F.mse_loss(pred_gz, gt_gz))

    return loss_g * gw


# ---- 预留损失函数（默认关闭） ----

def _compute_laplacian_loss(fld_pred, fld_gt, grid_shape, lw):
    """
    二阶空间梯度（拉普拉斯算子）损失 — 预留接口。
    TODO: 实现二阶差分并计算损失
    """
    raise NotImplementedError("Laplacian loss 尚未实现，请在配置中设置 laplacian: false")


def _compute_sharpness_loss(fld_pred, fld_gt_real, thresh, sw):
    """
    界面锐利度损失 — 预留接口。
    TODO: 鼓励界面区域的预测值趋向 0 或 1
    """
    raise NotImplementedError("Sharpness loss 尚未实现，请在配置中设置 sharpness: false")


# ---- 主训练损失函数 ----

def get_train_loss(fields, predict_hat, label_gt, normalizer, weight_cfg: dict, active_mask=None):
    """返回loss张量及监控指标（其余转为float）。"""
    num_channels = float(len(fields))

    losses = {
        "value_loss": torch.tensor(0),
        "grad_loss": torch.tensor(0),
        "loss": 0,
        'mean_l2': 0
        }

    pred_fp32 = predict_hat.float()
    label_fp32 = label_gt.float()
    with torch.no_grad():
        pred_real = normalizer.denormalize(pred_fp32)
        label_real = normalizer.denormalize(label_fp32)

    # ---- value loss ----
    weight_masks = {}  # 缓存各场的 weight_mask，供梯度损失复用

    if weight_cfg.get("enable", False):
        error_list = []
        weight_field = weight_cfg.get("field", [])
        thresholds = weight_cfg.get("threshold", [])
        bws = weight_cfg.get("base_weight", [])
        fws = weight_cfg.get("focus_weight", [])

        for idx, fld in enumerate(fields):
            fld_pred = pred_fp32[..., idx:idx+1]
            fld_gt = label_fp32[..., idx:idx+1]

            if fld in weight_field:
                i = weight_field.index(fld)
                thresh = thresholds[i]
                bw, fw = float(bws[i]), float(fws[i])
                fld_gt_real = label_real[..., idx:idx+1]
                is_vof = _is_vof_field(fld)

                loss_tensor, w_mask = _compute_weighted_value_loss(
                    fld_pred, fld_gt, fld_gt_real, thresh, bw, fw, is_vof
                )
                error_list.append(loss_tensor)
                weight_masks[fld] = w_mask
            else:
                error_list.append((fld_pred - fld_gt) ** 2)

        error_map = torch.cat(error_list, dim=-1)
        losses["value_loss"] = torch.mean(error_map)
    else:
        losses["value_loss"] = F.mse_loss(pred_fp32, label_fp32)

    # ---- gradient loss ----
    if weight_cfg.get("gradient", False):
        grad_loss_total = 0
        grad_weights = weight_cfg.get("grad_weight", {})
        grid_shape = weight_cfg.get("grid_shape", None)

        for idx, fld in enumerate(fields):
            if fld in grad_weights:
                gw = float(grad_weights[fld])
                is_vof = _is_vof_field(fld)
                fld_pred_g = pred_fp32[..., idx:idx+1]
                fld_gt_g = label_fp32[..., idx:idx+1]

                loss_g = _compute_gradient_loss(
                    fld_pred_g, fld_gt_g, grid_shape, gw,
                    is_vof, weight_cfg, weight_mask=weight_masks.get(fld)
                )
                grad_loss_total = grad_loss_total + loss_g

        losses["grad_loss"] = grad_loss_total

    # ---- 预留: laplacian / sharpness loss ----
    if weight_cfg.get("laplacian", False):
        lap_weights = weight_cfg.get("laplacian_weight", {})
        grid_shape = weight_cfg.get("grid_shape", None)
        lap_total = 0
        for idx, fld in enumerate(fields):
            if _is_vof_field(fld) and fld in lap_weights:
                lw = float(lap_weights[fld])
                lap_total = lap_total + _compute_laplacian_loss(
                    pred_fp32[..., idx:idx+1], label_fp32[..., idx:idx+1],
                    grid_shape, lw
                )
        losses["laplacian_loss"] = lap_total

    if weight_cfg.get("sharpness", False):
        sharp_weights = weight_cfg.get("sharpness_weight", {})
        thresholds = weight_cfg.get("threshold", [])
        weight_field = weight_cfg.get("field", [])
        sharp_total = 0
        for idx, fld in enumerate(fields):
            if _is_vof_field(fld) and fld in sharp_weights:
                sw = float(sharp_weights[fld])
                i = weight_field.index(fld) if fld in weight_field else None
                thresh = thresholds[i] if i is not None else [0.1, 0.9]
                sharp_total = sharp_total + _compute_sharpness_loss(
                    pred_fp32[..., idx:idx+1], label_real[..., idx:idx+1],
                    thresh, sw
                )
        losses["sharpness_loss"] = sharp_total

    # ---- 总 loss 合成 ----
    grad_multiplier = float(weight_cfg.get("grad_loss_multiplier", 8.0))
    losses["loss"] = losses["value_loss"] + grad_multiplier * losses["grad_loss"]

    if "laplacian_loss" in losses:
        lap_multiplier = float(weight_cfg.get("laplacian_loss_multiplier", 0.05))
        losses["loss"] = losses["loss"] + lap_multiplier * losses["laplacian_loss"]
    if "sharpness_loss" in losses:
        sharp_multiplier = float(weight_cfg.get("sharpness_loss_multiplier", 0.2))
        losses["loss"] = losses["loss"] + sharp_multiplier * losses["sharpness_loss"]

    # ---- 监控指标（不参与反向传播） ----
    with torch.no_grad():
        rmse = _rmse(pred_real, label_real)
        inactive_mask = (~active_mask) if active_mask is not None else None

        for i, fname in enumerate(fields):
            pred_ch_real = pred_real[..., i:i+1]
            gt_ch_real = label_real[..., i:i+1]

            rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)

            losses[f"L2_{fname}"] = rel_l2_val
            losses['mean_l2'] += rel_l2_val / num_channels
            losses[f"RMSE_{fname}"] = rmse[i].item()

            if active_mask is not None:
                ch_active = active_mask[..., i:i+1]
                ch_inactive = inactive_mask[..., i:i+1]
                losses[f"active_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_active)
                losses[f"inactive_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_inactive)
                losses[f"active_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_active)
                losses[f"inactive_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_inactive)

        if active_mask is not None:
            valid_active = [losses[f"active_L2_{f}"] for f in fields
                           if not math.isnan(losses.get(f"active_L2_{f}", float('nan')))]
            valid_inactive = [losses[f"inactive_L2_{f}"] for f in fields
                             if not math.isnan(losses.get(f"inactive_L2_{f}", float('nan')))]
            losses["active_mean_l2"] = sum(valid_active) / len(valid_active) if valid_active else float('nan')
            losses["inactive_mean_l2"] = sum(valid_inactive) / len(valid_inactive) if valid_inactive else float('nan')
            losses["active_loss"] = _masked_mse(pred_fp32, label_fp32, active_mask)
            losses["inactive_loss"] = _masked_mse(pred_fp32, label_fp32, inactive_mask)

    losses["each_l2"] = _each_l2(pred_real, label_real)

    return losses


def get_val_loss(fields, predict_hat, state, normalizer, active_mask=None):
    num_channels = float(len(fields))

    pred_fp32 = predict_hat.detach().float()
    label_fp32 = state.detach().float()

    pred_real = normalizer.denormalize(pred_fp32)
    state_real = normalizer.denormalize(label_fp32)

    losses = {
        'mean_l2': 0
        }

    rmse = _rmse(pred_real, state_real)
    inactive_mask = (~active_mask) if active_mask is not None else None

    for i, fname in enumerate(fields):
        pred_ch_real = pred_real[..., i:i+1]
        gt_ch_real = state_real[..., i:i+1]

        rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)

        losses[f"L2_{fname}"] = rel_l2_val
        losses['mean_l2'] += rel_l2_val / num_channels
        losses[f"RMSE_{fname}"] = rmse[i].item()

        if active_mask is not None:
            ch_active = active_mask[..., i:i+1]
            ch_inactive = inactive_mask[..., i:i+1]
            losses[f"active_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_active)
            losses[f"inactive_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_inactive)
            losses[f"active_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_active)
            losses[f"inactive_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_inactive)

    if active_mask is not None:
        valid_active = [losses[f"active_L2_{f}"] for f in fields
                       if not math.isnan(losses.get(f"active_L2_{f}", float('nan')))]
        valid_inactive = [losses[f"inactive_L2_{f}"] for f in fields
                         if not math.isnan(losses.get(f"inactive_L2_{f}", float('nan')))]
        losses["active_mean_l2"] = sum(valid_active) / len(valid_active) if valid_active else float('nan')
        losses["inactive_mean_l2"] = sum(valid_inactive) / len(valid_inactive) if valid_inactive else float('nan')

    losses["each_l2"] = _each_l2(pred_real, state_real)

    return losses


# ---- Region metric aggregation helpers ----

_REGION_PREFIXES = ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE")
_REGION_MEANS = ("active_mean_l2", "inactive_mean_l2")


def _init_region_agg(agg, fields):
    """Initialize region metric accumulators in the agg dict."""
    for fname in fields:
        for prefix in _REGION_PREFIXES:
            agg[f"{prefix}_{fname}"] = 0.0
            agg[f"{prefix}_{fname}_cnt"] = 0
    for key in _REGION_MEANS:
        agg[key] = 0.0
        agg[f"{key}_cnt"] = 0
    agg["active_loss"] = 0.0
    agg["inactive_loss"] = 0.0


def _accumulate_region(agg, costs, batch_num, fields, include_loss=True):
    """Accumulate region metrics from one batch, skipping NaN values."""
    for fname in fields:
        for prefix in _REGION_PREFIXES:
            val = costs.get(f"{prefix}_{fname}")
            if val is not None and not math.isnan(val):
                agg[f"{prefix}_{fname}"] += val * batch_num
                agg[f"{prefix}_{fname}_cnt"] += batch_num
    for key in _REGION_MEANS:
        val = costs.get(key)
        if val is not None and not math.isnan(val):
            agg[key] += val * batch_num
            agg[f"{key}_cnt"] += batch_num
    if include_loss:
        agg["active_loss"] += costs.get("active_loss", 0.0) * batch_num
        agg["inactive_loss"] += costs.get("inactive_loss", 0.0) * batch_num


def _finalize_region(agg, fields, total_num, include_loss=True):
    """Average region accumulators and remove count keys."""
    for fname in fields:
        for prefix in _REGION_PREFIXES:
            cnt = agg.pop(f"{prefix}_{fname}_cnt", 0)
            agg[f"{prefix}_{fname}"] = agg[f"{prefix}_{fname}"] / cnt if cnt > 0 else float('nan')
    for key in _REGION_MEANS:
        cnt = agg.pop(f"{key}_cnt", 0)
        agg[key] = agg[key] / cnt if cnt > 0 else float('nan')
    if include_loss:
        agg["active_loss"] = agg["active_loss"] / total_num if total_num > 0 else 0.0
        agg["inactive_loss"] = agg["inactive_loss"] / total_num if total_num > 0 else 0.0


def train(args, model, train_dataloader, optim, device, normalizer):
    horizon = args.data.get("horizon_train", 1) if isinstance(args.data, dict) else getattr(args, "horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp, check_point = args.train.get("use_amp", False), args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})

    agg = {}
    for key in ["loss", "L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0
    agg["value_loss"] = 0.0
    agg["grad_loss"] = 0.0
    has_region = False  # will be set True on first batch with active_mask

    model.train()
    normalizer.to(device)
    if use_amp: scaler = GradScaler('cuda')

    pbar = tqdm(train_dataloader, desc="  Train", unit="bt", leave=True, ncols=120, colour='green')
    for batch in pbar:
        dt = batch['dt'].to(device)
        state = batch["state"].to(device)  # [1 + horizon, N, 4]
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq = batch["time_seq"].to(device)
        conditions = batch["conditions"].to(device).float()
        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch['grid_shape'].numpy() # 针对一个 batch 生效

        active_mask = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask[:, 1:].to(device)  # slice off t=0 (input step)
            if not has_region:
                _init_region_agg(agg, fields)
                has_region = True

        batch_num = state.shape[0]

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)

            costs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()

        else:
            predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

            costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)
            costs["loss"].backward()
            optim.step()
            optim.zero_grad()

        agg["loss"] += costs["loss"].item() * batch_num
        agg["value_loss"] += costs["value_loss"].item() * batch_num
        agg["grad_loss"] += costs["grad_loss"].item() * batch_num

        for fname in fields:
            agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
            agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
        agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
        agg["each_l2"] += costs["each_l2"] * batch_num
        agg["num"] += batch_num

        if has_region:
            _accumulate_region(agg, costs, batch_num, fields, include_loss=True)

        avg_loss = agg["loss"] / agg["num"]
        pbar.set_postfix({
            "Loss": f"{avg_loss :.4e}"
        })

    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in ("active_loss", "inactive_loss") and key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=True)
    return agg


def validate(args, model, val_dataloader, device, normalizer, epoch):
    horizon = args.data.get("horizon_test", 1) if isinstance(args.data, dict) else getattr(args, "horizon_test", 1)
    fields = args.data.get("fields", ["T"])
    use_amp, check_point = args.train.get("use_amp", False), args.train.get("check_point", False)
    model_name = args.model.get("name", "PhysGTO")
    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3")
    agg = {}
    for key in ["L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0
    has_region = False

    model.eval()
    normalizer.to(device)

    # 随机选择样本进行可视化
    num_batches = len(val_dataloader)
    num_viz = min(2, num_batches)
    viz_batch_indices = set(random.sample(range(num_batches), num_viz))

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="  Valid", unit="bt", leave=False, ncols=120, colour='yellow')
        for i, batch in enumerate(pbar):
            dt = batch['dt'].to(device)
            state = batch["state"].to(device)
            node_pos = batch["node_pos"].to(device)
            edges = batch["edges"].to(device)
            time_seq = batch["time_seq"].to(device)
            if _use_spatial:
                spatial_inform = batch["spatial_inform"].to(device)
            conditions = batch["conditions"].to(device).float()

            active_mask = batch.get("active_mask")
            if active_mask is not None:
                active_mask = active_mask[:, 1:].to(device)
                if not has_region:
                    _init_region_agg(agg, fields)
                    has_region = True

            batch_num = state.shape[0]

            if use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    if _use_spatial:
                        predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
                    else:
                        predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                    costs = get_val_loss(fields, predict_hat, state[:, 1:], normalizer, active_mask=active_mask)
            else:
                if _use_spatial:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
                else:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                costs = get_val_loss(fields, predict_hat, state[:, 1:], normalizer, active_mask=active_mask)

            for fname in fields:
                agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
                agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
            agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
            agg["each_l2"] += costs["each_l2"] * batch_num
            agg["num"] += batch_num

            if has_region:
                _accumulate_region(agg, costs, batch_num, fields, include_loss=False)


    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=False)
    return agg
