import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from pathlib import Path

VELOCITY_FIELD_NAMES = ("Ux", "Uy", "Uz")


def _magnitude_aware_loss(pred, target, mask=None, base_weight=1.0, mag_scale=5.0):
    """
    Loss that weights each node by the velocity magnitude of the ground truth.
    Uses a FIXED reference scale (99th percentile per-sample) instead of batch max,
    which stabilizes training by avoiding batch-dependent loss scaling.
    """
    # Per-node velocity magnitude of ground truth: [B,T,N,1]
    gt_mag = torch.norm(target, dim=-1, keepdim=True)
    # Use 99th percentile per sample (more robust than max, avoids outlier sensitivity)
    # Flatten spatial+time dims, compute quantile per batch element
    B = gt_mag.shape[0]
    flat_mag = gt_mag.reshape(B, -1)
    ref_scale = torch.quantile(flat_mag, 0.99, dim=1, keepdim=True).clamp_min(1e-6)
    ref_scale = ref_scale.view(B, 1, 1, 1)

    # Normalized magnitude weight
    w = base_weight + mag_scale * (gt_mag / ref_scale).clamp_max(1.0)

    # Huber loss with smaller beta for better gradient signal on small errors
    pointwise_loss = F.smooth_l1_loss(pred, target, reduction='none', beta=0.05)

    weighted = pointwise_loss * w
    if mask is not None:
        weighted = weighted * mask
        return weighted.sum() / mask.sum().clamp_min(1.0)
    return weighted.mean()


def _spatial_gradient_loss(pred, target, edges, mask=None):
    """
    Penalizes differences in spatial gradients between prediction and ground truth.
    Uses Huber loss instead of MSE for robustness to sharp gradient outliers.
    """
    # pred, target: [B, T, N, C]
    B, T, N, C = pred.shape

    send_idx = edges[..., 0].long()  # [B, E]
    recv_idx = edges[..., 1].long()  # [B, E]
    E_edges = send_idx.shape[1]

    # Expand indices for gathering: [B, T, E, C]
    send_exp = send_idx[:, None, :, None].expand(B, T, E_edges, C)
    recv_exp = recv_idx[:, None, :, None].expand(B, T, E_edges, C)

    pred_send = torch.gather(pred, 2, send_exp)
    pred_recv = torch.gather(pred, 2, recv_exp)
    gt_send = torch.gather(target, 2, send_exp)
    gt_recv = torch.gather(target, 2, recv_exp)

    pred_grad = pred_recv - pred_send   # [B, T, E, C]
    gt_grad = gt_recv - gt_send

    # Use Huber loss for robustness at sharp interfaces
    grad_err = F.smooth_l1_loss(pred_grad, gt_grad, reduction='none', beta=0.1)

    if mask is not None:
        # Build edge-level mask from node-level mask
        mask_send = torch.gather(mask, 2, send_exp)
        mask_recv = torch.gather(mask, 2, recv_exp)
        edge_mask = mask_send * mask_recv
        grad_err = grad_err * edge_mask
        return grad_err.sum() / edge_mask.sum().clamp_min(1.0)

    return grad_err.mean()


def _temporal_consistency_loss(pred, target, mask=None):
    """
    Penalizes temporal inconsistency: the predicted temporal difference
    should match the ground truth temporal difference.
    pred, target: [B, T, N, C]
    """
    if pred.shape[1] < 2:
        return pred.new_zeros(())

    pred_dt = pred[:, 1:] - pred[:, :-1]    # [B, T-1, N, C]
    gt_dt = target[:, 1:] - target[:, :-1]

    err = F.smooth_l1_loss(pred_dt, gt_dt, reduction='none', beta=0.05)

    if mask is not None:
        mask_t = mask[:, 1:]  # [B, T-1, N, C]
        err = err * mask_t
        return err.sum() / mask_t.sum().clamp_min(1.0)
    return err.mean()


def _velocity_direction_loss(pred, target, mask=None, mag_threshold=0.01):
    """
    Cosine similarity loss for velocity direction.
    Only applies where velocity magnitude is significant (inside melt pool).
    Ensures the model gets flow DIRECTION right, not just magnitude.
    """
    # pred, target: [B, T, N, C] where C = 3 (Ux, Uy, Uz)
    gt_mag = torch.norm(target, dim=-1, keepdim=True)  # [B, T, N, 1]

    # Only penalize direction where velocity is significant
    active = (gt_mag > mag_threshold).squeeze(-1)  # [B, T, N]

    if mask is not None:
        # Combine with existing mask (all channels must be active)
        mask_all = mask.min(dim=-1).values > 0.5  # [B, T, N]
        active = active & mask_all

    if active.sum() < 10:
        return pred.new_zeros(())

    # Cosine similarity on active nodes
    pred_active = pred[active]   # [K, C]
    target_active = target[active]  # [K, C]

    cos_sim = F.cosine_similarity(pred_active, target_active, dim=-1)  # [K]
    # Loss = 1 - cos_sim (0 when perfect, 2 when opposite direction)
    return (1.0 - cos_sim).mean()


# l2 误差计算需要反归一化数据
def _relative_l2(pred, target, mask=None):
    """相对L2误差，返回 [batch] 张量。"""
    if mask is not None:
        pred, target = pred * mask, target * mask
    error = pred - target
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-6)
    norm_error_channel = torch.mean(norm_error, dim=-1)  # 平均所有通道
    norm_error_time = torch.mean(norm_error_channel, dim=-1)
    return norm_error_time

def _each_l2(pred, target, mask=None):
    t_step = target.shape[1]
    losses_each_t = torch.zeros(t_step, device=pred.device)
    for t in range(t_step):
        pred_t = pred[:, t]
        target_t = target[:, t]
        if mask is not None:
            mask_t = mask[:, t]
            pred_t = pred_t * mask_t
            target_t = target_t * mask_t
        error = pred_t - target_t
        norm_error = torch.norm(error, dim=-2) / (torch.norm(target_t, dim=-2) + 1e-6)
        norm_error_channel = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_channel, dim=0)
        losses_each_t[t] = norm_error_batch
    return losses_each_t

# rmse 同样需要反归一化数据
def _rmse(pred, target, mask=None):
    diff = pred - target
    if mask is not None:
        mask = mask.to(diff.dtype)
        mse = torch.sum((diff ** 2) * mask, dim=[0, 1, 2]) / torch.clamp_min(mask.sum(dim=[0, 1, 2]), 1.0)
    else:
        mse = torch.mean(diff**2, dim=[0, 1, 2])
    return torch.sqrt(mse)

def _get_velocity_axis_info(model, fields): # 获取速度场的信息（索引）
    if model is not None and hasattr(model, "get_velocity_axis_info"):
        return model.get_velocity_axis_info(fields)

    axis_info = []
    for axis_id, field_name in enumerate(VELOCITY_FIELD_NAMES):
        if field_name in fields:
            axis_info.append((axis_id, field_name, fields.index(field_name)))
    return axis_info

def _compute_velocity_divergence(velocity, node_pos, edges, axis_info, node_mask=None, eps=1e-10):
    """
    根据图结构对速度场做有限差分，近似散度
    velocity: [B, T, N, C_vel]，通道顺序需与 axis_info 一致
    node_pos: [B, N, 3] 物理坐标
    edges:    [B, E, 2]
    node_mask:[B, T, N]，True 表示该节点参与散度约束
    """
    batch_size, horizon, num_nodes, vel_dim = velocity.shape
    num_edges = edges.shape[1]
    space_dim = node_pos.shape[-1]

    send_idx = edges[..., 0].long()
    recv_idx = edges[..., 1].long()

    sender_pos = torch.gather(node_pos, 1, send_idx.unsqueeze(-1).expand(-1, -1, space_dim))
    receiver_pos = torch.gather(node_pos, 1, recv_idx.unsqueeze(-1).expand(-1, -1, space_dim))
    delta_pos = receiver_pos - sender_pos
    abs_delta = delta_pos.abs()

    edge_axis = abs_delta.argmax(dim=-1) # 位移最大的轴
    spacing = abs_delta.gather(-1, edge_axis.unsqueeze(-1)).squeeze(-1).clamp_min(eps) # 沿主轴的位移量

    axis_lookup = torch.full((space_dim,), -1, dtype=torch.long, device=edges.device)
    for vel_slot, (axis_id, _, _) in enumerate(axis_info):
        axis_lookup[axis_id] = vel_slot
    vel_slot = axis_lookup[edge_axis]
    supported_edge = vel_slot >= 0
    safe_vel_slot = vel_slot.clamp_min(0) # 将 -1 clamp 到 0，防止 gather 越界

    sender_vel = torch.gather(
        velocity,
        2,
        send_idx[:, None, :, None].expand(-1, horizon, -1, vel_dim)
    )
    receiver_vel = torch.gather(
        velocity,
        2,
        recv_idx[:, None, :, None].expand(-1, horizon, -1, vel_dim)
    )
    vel_slot_t = safe_vel_slot[:, None, :, None].expand(-1, horizon, -1, 1)
    sender_axis_vel = torch.gather(sender_vel, -1, vel_slot_t).squeeze(-1)
    receiver_axis_vel = torch.gather(receiver_vel, -1, vel_slot_t).squeeze(-1)

    edge_grad = (receiver_axis_vel - sender_axis_vel) / spacing[:, None, :]

    if node_mask is None:
        edge_mask = supported_edge[:, None, :].expand(-1, horizon, -1)
    else:
        send_valid = torch.gather(node_mask, 2, send_idx[:, None, :].expand(-1, horizon, -1))
        recv_valid = torch.gather(node_mask, 2, recv_idx[:, None, :].expand(-1, horizon, -1))
        edge_mask = send_valid & recv_valid & supported_edge[:, None, :]

    edge_weight = edge_mask.to(edge_grad.dtype)
    edge_grad = edge_grad * edge_weight

    flat_bt = batch_size * horizon
    edge_grad_flat = edge_grad.reshape(flat_bt, num_edges)
    edge_weight_flat = edge_weight.reshape(flat_bt, num_edges)
    vel_slot_flat = safe_vel_slot[:, None, :].expand(-1, horizon, -1).reshape(flat_bt, num_edges)
    send_flat = send_idx[:, None, :].expand(-1, horizon, -1).reshape(flat_bt, num_edges)
    recv_flat = recv_idx[:, None, :].expand(-1, horizon, -1).reshape(flat_bt, num_edges)

    sender_slot = send_flat * vel_dim + vel_slot_flat
    receiver_slot = recv_flat * vel_dim + vel_slot_flat

    grad_sum = edge_grad.new_zeros(flat_bt, num_nodes * vel_dim)
    grad_count = edge_grad.new_zeros(flat_bt, num_nodes * vel_dim)
    grad_sum.scatter_add_(1, sender_slot, edge_grad_flat)
    grad_sum.scatter_add_(1, receiver_slot, edge_grad_flat)
    grad_count.scatter_add_(1, sender_slot, edge_weight_flat)
    grad_count.scatter_add_(1, receiver_slot, edge_weight_flat)

    mean_grad = grad_sum / grad_count.clamp_min(1.0)
    mean_grad = mean_grad.view(batch_size, horizon, num_nodes, vel_dim)
    divergence = mean_grad.sum(dim=-1)

    if node_mask is not None:
        divergence = divergence * node_mask.to(divergence.dtype)

    return divergence, spacing, edge_mask

def get_incompressibility_loss(args, model, predict_hat, label_gt, normalizer, fields, node_pos_phys, edges, node_type=None, mask_weight=None, epoch=0):
    train_args = args.train
    cfg = train_args.get("incompressibility", {}) or {}
    axis_info = _get_velocity_axis_info(model, fields)

    expected_axes = min(node_pos_phys.shape[-1], len(VELOCITY_FIELD_NAMES))
    enabled = cfg.get("enabled")
    if enabled is None:
        enabled = len(axis_info) >= expected_axes and expected_axes >= 2

    zero = predict_hat.new_zeros(())
    if not enabled or len(axis_info) < expected_axes:
        return {
            "loss": zero,
            "weighted_loss": zero,
            "rms": zero,
            "scaled_rms": zero,
        }

    pred_phys = normalizer.denormalize(predict_hat)
    gt_phys = normalizer.denormalize(label_gt)
    vel_indices = [field_idx for _, _, field_idx in axis_info]
    pred_vel = torch.cat([pred_phys[..., idx:idx+1] for idx in vel_indices], dim=-1)
    gt_vel = torch.cat([gt_phys[..., idx:idx+1] for idx in vel_indices], dim=-1)

    if mask_weight is None:
        node_mask = torch.ones_like(pred_vel[..., 0], dtype=torch.bool)
    else:
        vel_mask = torch.cat([mask_weight[..., idx:idx+1] for idx in vel_indices], dim=-1)
        node_mask = torch.all(vel_mask > 0.5, dim=-1)

    if node_type is not None and cfg.get("interior_only", True):
        node_mask = node_mask & (node_type.squeeze(-1) == 0).unsqueeze(1)

    pred_div, spacing, edge_mask = _compute_velocity_divergence(pred_vel, node_pos_phys, edges, axis_info, node_mask=node_mask)
    target_mode = cfg.get("target", "zero") # 散度目标

    if target_mode == "gt":
        div_target, _, _ = _compute_velocity_divergence(gt_vel, node_pos_phys, edges, axis_info, node_mask=node_mask)
    else:
        div_target = torch.zeros_like(pred_div)

    div_residual = pred_div - div_target
    node_weight = node_mask.to(pred_div.dtype)
    node_count = node_weight.sum().clamp_min(1.0)

    speed_mag = torch.linalg.norm(gt_vel, dim=-1)
    speed_scale = (speed_mag * node_weight).sum() / node_count
    speed_scale = speed_scale.clamp_min(float(cfg.get("speed_floor", 1e-6)))

    edge_weight = edge_mask.to(spacing.dtype)
    edge_count = edge_weight.sum().clamp_min(1.0)
    length_scale = (spacing[:, None, :] * edge_weight).sum() / edge_count
    length_scale = length_scale.clamp_min(float(cfg.get("length_floor", 1e-9)))

    div_scaled = div_residual * (length_scale / speed_scale)
    div_loss = torch.sum((div_scaled ** 2) * node_weight) / node_count

    # warm-up 调度：前 warmup_epochs 个 epoch 线性增长权重
    base_weight = float(cfg.get("weight", 5e-2))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        ramp = (epoch + 1) / warmup_epochs
        weight = base_weight * ramp
    else:
        weight = base_weight
    weighted_div_loss = div_loss * weight

    div_rms = torch.sqrt(torch.sum((div_residual ** 2) * node_weight) / node_count)
    div_scaled_rms = torch.sqrt(torch.sum((div_scaled ** 2) * node_weight) / node_count)

    return {
        "loss": div_loss,
        "weighted_loss": weighted_div_loss,
        "rms": div_rms.detach(),
        "scaled_rms": div_scaled_rms.detach(),
    }

def get_weighted_mse_loss(predict_hat, label_gt, normalizer, fields, train_args, device, mask_weight=None):
    """
    计算加权 MSE Loss
    predict_hat, label_gt: 归一化后的 Tensor [B, T, N, C]
    """
    weight_cfg = train_args.get("loss_weight", None)
    calc_mode = train_args.get("calc_mode", "mse_norm")

    if weight_cfg is None or not weight_cfg.get("enabled", False):
        if calc_mode == "mse_norm":
            return F.mse_loss(predict_hat, label_gt, reduction='mean')
        elif calc_mode == "rel_phys":
            pred_norm, gt_norm = predict_hat, label_gt
            pred_phys, gt_phys = normalizer.denormalize(predict_hat), normalizer.denormalize(label_gt)
            error_list = []
            for i, fname in enumerate(fields):
                if fname == "T":
                    p, g = pred_phys[..., i:i+1], gt_phys[..., i:i+1]
                    err = ((p - g) / (torch.abs(g) + 1e-6)) ** 2
                else:
                    p, g = pred_norm[..., i:i+1], gt_norm[..., i:i+1]
                    err = (p - g) ** 2

                error_list.append(err)
            return torch.mean(torch.cat(error_list, dim=-1))

    if calc_mode == "mse_norm":
        pred_val, gt_val = predict_hat, label_gt
        base_error_map = (pred_val - gt_val) ** 2
        gt_denorm = normalizer.denormalize(label_gt)
    elif calc_mode == "rel_phys":
        pred_phys, gt_phys = normalizer.denormalize(predict_hat), normalizer.denormalize(label_gt)
        pred_norm, gt_norm = predict_hat, label_gt
        gt_denorm = gt_phys
        base_error_map = ((pred_phys - gt_phys) / (torch.abs(gt_phys) + 1e-6)) ** 2

        error_list = []
        for i, fname in enumerate(fields):
            if fname == "T":
                p, g = pred_phys[..., i:i+1], gt_phys[..., i:i+1]
                err = ((p - g) / (torch.abs(g) + 1e-6)) ** 2
            else:
                p, g = pred_norm[..., i:i+1], gt_norm[..., i:i+1]
                err = (p - g) ** 2

            error_list.append(err)
        base_error_map = torch.cat(error_list, dim=-1)

    weights = torch.full_like(label_gt[..., :1], weight_cfg.get("bg_weight", 1.0))

    target_field = weight_cfg.get("target_field", "T")

    if target_field in fields:
        f_idx = fields.index(target_field)

        gt_target_phys = gt_denorm[..., f_idx:f_idx+1]

        threshold_val = weight_cfg.get("threshold_val", 500.0)
        mask = gt_target_phys > threshold_val

        weights[mask] = weight_cfg.get("high_weight", 10.0)

    weighted_sq_error = base_error_map * weights

    if mask_weight is not None:
        channel_losses = []
        for i, field in enumerate(fields):
            err_ch = weighted_sq_error[..., i]  # 当前通道的误差 [B, T, N]
            mask_ch = mask_weight[..., i]        # 当前通道的掩码 [B, T, N]
            sum_err = torch.sum(err_ch * mask_ch) # 分子
            valid_nodes = torch.sum(mask_ch)  # 分母
            if valid_nodes > 0: loss_ch = sum_err / valid_nodes
            else: loss_ch = sum_err * 0.0
            channel_losses.append(loss_ch)
        loss = torch.mean(torch.stack(channel_losses))
    else:
        loss = torch.mean(weighted_sq_error)
    return loss

def get_train_loss(args, model, predict_hat, label_gt, normalizer, node_pos_phys=None, edges=None, node_type=None, mask_weight=None, epoch=0):
    """返回loss张量及监控指标（其余转为float）。"""
    train_args = args.train
    fields = args.data.get("fields", ["T"])

    # === Primary loss: magnitude-aware Huber ===
    mag_scale = float(train_args.get("mag_scale", 5.0))
    data_loss = _magnitude_aware_loss(predict_hat, label_gt, mask=mask_weight, mag_scale=mag_scale)

    # === Spatial gradient loss ===
    grad_weight = float(train_args.get("grad_loss_weight", 0.1))
    grad_warmup = int(train_args.get("grad_loss_warmup", 5))
    if grad_weight > 0 and edges is not None:
        grad_loss = _spatial_gradient_loss(predict_hat, label_gt, edges, mask=mask_weight)
        # Warm up the gradient loss
        if grad_warmup > 0 and epoch < grad_warmup:
            grad_ramp = (epoch + 1) / grad_warmup
        else:
            grad_ramp = 1.0
        grad_loss_weighted = grad_weight * grad_ramp * grad_loss
    else:
        grad_loss = predict_hat.new_zeros(())
        grad_loss_weighted = predict_hat.new_zeros(())

    # === Temporal consistency loss ===
    temp_weight = float(train_args.get("temporal_loss_weight", 0.05))
    if temp_weight > 0 and predict_hat.shape[1] >= 2:
        temp_loss = _temporal_consistency_loss(predict_hat, label_gt, mask=mask_weight)
        temp_loss_weighted = temp_weight * temp_loss
    else:
        temp_loss = predict_hat.new_zeros(())
        temp_loss_weighted = predict_hat.new_zeros(())

    # === Velocity direction (cosine) loss ===
    dir_weight = float(train_args.get("direction_loss_weight", 0.1))
    dir_warmup = int(train_args.get("direction_loss_warmup", 5))
    vel_fields = [f for f in fields if f in VELOCITY_FIELD_NAMES]
    if dir_weight > 0 and len(vel_fields) == 3:
        vel_indices = [fields.index(f) for f in vel_fields]
        pred_vel = torch.cat([predict_hat[..., i:i+1] for i in vel_indices], dim=-1)
        gt_vel = torch.cat([label_gt[..., i:i+1] for i in vel_indices], dim=-1)
        vel_mask = torch.cat([mask_weight[..., i:i+1] for i in vel_indices], dim=-1) if mask_weight is not None else None
        dir_loss = _velocity_direction_loss(pred_vel, gt_vel, mask=vel_mask)
        if dir_warmup > 0 and epoch < dir_warmup:
            dir_ramp = (epoch + 1) / dir_warmup
        else:
            dir_ramp = 1.0
        dir_loss_weighted = dir_weight * dir_ramp * dir_loss
    else:
        dir_loss = predict_hat.new_zeros(())
        dir_loss_weighted = predict_hat.new_zeros(())

    # === Incompressibility loss ===
    incompressibility = get_incompressibility_loss(
        args,
        model,
        predict_hat,
        label_gt,
        normalizer,
        fields,
        node_pos_phys=node_pos_phys,
        edges=edges,
        node_type=node_type,
        mask_weight=mask_weight,
        epoch=epoch,
    ) if node_pos_phys is not None and edges is not None else {
        "loss": predict_hat.new_zeros(()),
        "weighted_loss": predict_hat.new_zeros(()),
        "rms": predict_hat.new_zeros(()),
        "scaled_rms": predict_hat.new_zeros(()),
    }

    loss_val = data_loss + grad_loss_weighted + temp_loss_weighted + dir_loss_weighted + incompressibility["weighted_loss"]

    num_channels = float(len(fields))

    losses = {
        "loss": loss_val,
        "data_loss": data_loss.detach(),
        "grad_loss": grad_loss.detach(),
        "temp_loss": temp_loss.detach(),
        "dir_loss": dir_loss.detach(),
        "div_loss": incompressibility["weighted_loss"].detach(),
        "div_raw_loss": incompressibility["loss"].detach(),
        "div_rms": incompressibility["rms"],
        "div_scaled_rms": incompressibility["scaled_rms"],
        'mean_l2': 0
        }

    with torch.no_grad():
        pred_real = normalizer.denormalize(predict_hat)
        label_real = normalizer.denormalize(label_gt)
        rmse = _rmse(pred_real, label_real, mask=mask_weight)

        for i, fname in enumerate(fields):
            pred_ch_real = pred_real[..., i:i+1]
            gt_ch_real = label_real[..., i:i+1]
            field_mask = mask_weight[..., i:i+1] if mask_weight is not None else None

            rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real, mask=field_mask)

            losses[f"L2_{fname}"] = rel_l2_val
            losses['mean_l2'] += rel_l2_val / num_channels
            losses[f"RMSE_{fname}"] = rmse[i].item()

    losses["each_l2"] = _each_l2(pred_real, label_real, mask=mask_weight)

    return losses

def get_val_loss(args, model, fields, predict_hat, state, normalizer, node_pos_phys=None, edges=None, node_type=None, mask_weight=None):
    num_channels = float(len(fields))

    pred_real = normalizer.denormalize(predict_hat)
    state_real = normalizer.denormalize(state)

    losses = {
        'mean_l2': 0
        }

    rmse = _rmse(pred_real, state_real, mask=mask_weight)

    for i, fname in enumerate(fields):
        pred_ch_real = pred_real[..., i:i+1]
        gt_ch_real = state_real[..., i:i+1]
        field_mask = mask_weight[..., i:i+1] if mask_weight is not None else None

        rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real, mask=field_mask)

        losses[f"L2_{fname}"] = rel_l2_val
        losses['mean_l2'] += rel_l2_val / num_channels
        losses[f"RMSE_{fname}"] = rmse[i].item()

    losses["each_l2"] = _each_l2(pred_real, state_real, mask=mask_weight)

    incompressibility = get_incompressibility_loss(
        args,
        model,
        predict_hat,
        state,
        normalizer,
        fields,
        node_pos_phys=node_pos_phys,
        edges=edges,
        node_type=node_type,
        mask_weight=mask_weight,
    ) if node_pos_phys is not None and edges is not None else {
        "loss": predict_hat.new_zeros(()),
        "weighted_loss": predict_hat.new_zeros(()),
        "rms": predict_hat.new_zeros(()),
        "scaled_rms": predict_hat.new_zeros(()),
    }
    losses["div_raw_loss"] = incompressibility["loss"].detach()
    losses["div_rms"] = incompressibility["rms"]
    losses["div_scaled_rms"] = incompressibility["scaled_rms"]

    return losses


def train(args, model, train_dataloader, optim, device, normalizer, epoch=0):
    horizon = args.data.get("horizon_train", 1) if isinstance(args.data, dict) else getattr(args, "horizon_train", 1)
    fields, data_mask = args.data.get("fields", ["T"]), args.data.get("mask", False)

    # Teacher forcing with curriculum decay
    teacher_cfg = args.train.get("teacher", False)
    decay_start = int(args.train.get("teacher_decay_start", 30))
    decay_end = int(args.train.get("teacher_decay_end", 100))
    if teacher_cfg:
        if epoch < decay_start:
            teacher_prob = 1.0
        elif epoch < decay_end:
            teacher_prob = 1.0 - (epoch - decay_start) / (decay_end - decay_start)
        else:
            teacher_prob = 0.0
        teacher = random.random() < teacher_prob
    else:
        teacher = False
    agg = {}
    for key in ["loss", "data_loss", "grad_loss", "temp_loss", "dir_loss", "div_loss", "div_raw_loss", "div_rms", "div_scaled_rms", "L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0

    model.train()
    normalizer.to(device)

    pbar = tqdm(train_dataloader, desc="  Train", unit="bt", leave=True, ncols=120, colour='green')
    for batch in pbar:
        dt = batch['dt'].to(device)
        state = batch["state"].to(device)  # [1 + horizon, N, 4]
        node_pos = batch["node_pos"].to(device)
        node_pos_phys = batch.get("node_pos_phys", batch["node_pos"]).to(device)
        edges = batch["edges"].to(device)
        node_type = batch["node_type"].to(device)
        time_seq = batch["time_seq"].to(device)
        conditions = batch["conditions"].to(device).float()

        batch_num = state.shape[0]

        predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, teacher, state[:, 1:])

        valid_mask = torch.ones_like(predict_hat)

        if data_mask:
            T = predict_hat.shape[1]
            gas_mask, y_cutoff_mask, solid_mask = batch["gas_mask"].to(device), batch["y_cutoff_mask"].to(device), batch["solid_mask"].to(device)
            gas_mask_bool, solid_mask_bool = gas_mask[:, 1:, :], solid_mask[:, 1:, :]
            vel_inactive_mask = gas_mask_bool | solid_mask_bool
            if len(y_cutoff_mask.shape) == 3:# [B, N, 1]
                y_mask_bool = y_cutoff_mask.squeeze(-1).unsqueeze(1).expand(-1, T, -1)
            else:
                y_mask_bool = y_cutoff_mask.unsqueeze(1).expand(-1, T, -1)

            for i, field in enumerate(fields):
                if field in ["Ux", "Uy", "Uz"]:
                    state[:, 1:, :, i][vel_inactive_mask] = 0.0
                    predict_hat[..., i][vel_inactive_mask] = 0.0
                    valid_mask[..., i][vel_inactive_mask] = 0.0 # 计算 loss 的时候忽略这些区域
                elif field in ["gamma_liquid"]:
                    state[:, 1:, :, i][y_mask_bool] = 0.0
                    predict_hat[..., i][y_mask_bool] = 0.0
                    valid_mask[..., i][y_mask_bool] = 0.0

        costs = get_train_loss(
            args,
            model,
            predict_hat,
            state[:, 1:],
            normalizer,
            node_pos_phys=node_pos_phys,
            edges=edges,
            node_type=node_type,
            mask_weight=valid_mask,
            epoch=epoch,
        )

        optim.zero_grad(set_to_none=True)
        costs["loss"].backward()

        grad_clip = args.train.get("grad_clip", None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optim.step()


        agg["loss"] += costs["loss"].item() * batch_num
        agg["data_loss"] += costs["data_loss"].item() * batch_num
        agg["grad_loss"] += costs["grad_loss"].item() * batch_num
        agg["temp_loss"] += costs["temp_loss"].item() * batch_num
        agg["dir_loss"] += costs["dir_loss"].item() * batch_num
        agg["div_loss"] += costs["div_loss"].item() * batch_num
        agg["div_raw_loss"] += costs["div_raw_loss"].item() * batch_num
        agg["div_rms"] += costs["div_rms"].item() * batch_num
        agg["div_scaled_rms"] += costs["div_scaled_rms"].item() * batch_num
        for fname in fields:
            agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
            agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
        agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
        agg["each_l2"] += costs["each_l2"] * batch_num
        agg["num"] += batch_num

        avg_loss = agg["loss"] / agg["num"]
        pbar.set_postfix({
            "Loss": f"{avg_loss :.4e}"
        })

    for key, value in agg.items():
        if key != "each_l2" and key != "num":
            agg[key] = value / agg["num"]
    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    return agg


def validate(args, model, val_dataloader, device, normalizer, epoch):
    horizon = args.data.get("horizon_test", 1) if isinstance(args.data, dict) else getattr(args, "horizon_test", 1)
    fields, data_mask = args.data.get("fields", ["T"]), args.data.get("mask", False)
    agg = {}
    for key in ["L2", "mean_l2", "RMSE", "div_raw_loss", "div_rms", "div_scaled_rms"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0

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
            node_pos_phys = batch.get("node_pos_phys", batch["node_pos"]).to(device)
            edges = batch["edges"].to(device)
            node_type = batch["node_type"].to(device)
            time_seq = batch["time_seq"].to(device)
            conditions = batch["conditions"].to(device).float()

            batch_num = state.shape[0]
            predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt)

            valid_mask = torch.ones_like(predict_hat)

            if data_mask:
                T = predict_hat.shape[1]
                gas_mask, y_cutoff_mask, solid_mask = batch["gas_mask"].to(device), batch["y_cutoff_mask"].to(device), batch["solid_mask"].to(device)
                gas_mask_bool, solid_mask_bool = gas_mask[:, 1:, :], solid_mask[:, 1:, :]
                vel_inactive_mask = gas_mask_bool | solid_mask_bool
                if len(y_cutoff_mask.shape) == 3:# [B, N, 1]
                    y_mask_bool = y_cutoff_mask.squeeze(-1).unsqueeze(1).expand(-1, T, -1)
                else:
                    y_mask_bool = y_cutoff_mask.unsqueeze(1).expand(-1, T, -1)

                for i, field in enumerate(fields):
                    if field in ["Ux", "Uy", "Uz"]:
                        state[:, 1:, :, i][vel_inactive_mask] = 0.0
                        predict_hat[..., i][vel_inactive_mask] = 0.0
                        valid_mask[..., i][vel_inactive_mask] = 0.0 # 计算 loss 的时候忽略这些区域
                    elif field in ["gamma_liquid"]:
                        state[:, 1:, :, i][y_mask_bool] = 0.0
                        predict_hat[..., i][y_mask_bool] = 0.0
                        valid_mask[..., i][y_mask_bool] = 0.0

            costs = get_val_loss(
                args,
                model,
                fields,
                predict_hat,
                state[:, 1:],
                normalizer,
                node_pos_phys=node_pos_phys,
                edges=edges,
                node_type=node_type,
                mask_weight=valid_mask,
            )

            for fname in fields:
                agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
                agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
            agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
            agg["div_raw_loss"] += costs["div_raw_loss"].item() * batch_num
            agg["div_rms"] += costs["div_rms"].item() * batch_num
            agg["div_scaled_rms"] += costs["div_scaled_rms"].item() * batch_num
            agg["each_l2"] += costs["each_l2"] * batch_num
            agg["num"] += batch_num

    for key, value in agg.items():
        if key != "each_l2" and key != "num":
            agg[key] = value / agg["num"]
    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    return agg
