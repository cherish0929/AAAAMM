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

def get_train_loss(fields, predict_hat, label_gt, normalizer, weight_cfg: dict):
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

    if weight_cfg.get("enable", False):
        error_list = []
        weight_field = weight_cfg.get("field")
        thresholds = weight_cfg.get("threshold", [])
        bws = weight_cfg.get("base_weight", [])
        fws = weight_cfg.get("focus_weight", [])
        for idx, fld in enumerate(fields):
            fld_pred, fld_gt = pred_fp32[..., idx:idx+1], label_fp32[..., idx:idx+1]
            squared_error = (fld_pred - fld_gt) ** 2

            if fld in weight_field:
                i = weight_field.index(fld)
                thresh, bw, fw = thresholds[i], float(bws[i]), float(fws[i])
                fld_gt_real = label_real[..., idx:idx+1] # [B, T, N, 1]
                with torch.no_grad():
                    if isinstance(thresh, list):
                        condition = (fld_gt_real > thresh[0]) & (fld_gt_real < thresh[1])
                        weight_mask = torch.where(condition, fw, bw)
                    else:
                        weight_mask = torch.where(fld_gt_real > thresh, fw, bw)
                    
                    weight_mask = weight_mask / torch.mean(weight_mask)
                
                weighted_squared_error = squared_error * weight_mask
                error_list.append(weighted_squared_error)
            else:
                error_list.append(squared_error)
            
        error_map = torch.cat(error_list, dim=-1)
        losses["value_loss"] = torch.mean(error_map)
        
    else:
        losses["value_loss"] = F.mse_loss(pred_fp32, label_fp32)

    if weight_cfg.get("gradient", False):
        grad_loss_total = 0
        grad_weights = weight_cfg.get("grad_weight", {})
        grid_shape = weight_cfg.get("grid_shape", None)
        for idx, fld in enumerate(fields):
            if fld in grad_weights:
                gw = float(grad_weights[fld])
                fld_pred_g = pred_fp32[..., idx:idx+1]
                fld_gt_g = label_fp32[..., idx:idx+1]
                pred_gx, pred_gy, pred_gz = compute_spatial_gradient_3d(fld_pred_g, grid_shape)
                gt_gx, gt_gy, gt_gz       = compute_spatial_gradient_3d(fld_gt_g, grid_shape)
                # 梯度的纯 MSE
                loss_gx = F.mse_loss(pred_gx, gt_gx, reduction='mean')
                loss_gy = F.mse_loss(pred_gy, gt_gy, reduction='mean')
                loss_gz = F.mse_loss(pred_gz, gt_gz, reduction='mean')

                loss_g = loss_gx + loss_gy + loss_gz
                grad_loss_total += loss_g * gw
        losses["grad_loss"] = grad_loss_total
        
    losses["loss"] = losses["value_loss"] + 8.0 * losses["grad_loss"]

    with torch.no_grad():
        rmse = _rmse(pred_real, label_real)

        for i, fname in enumerate(fields):
            pred_ch_real = pred_real[..., i:i+1]
            gt_ch_real = label_real[..., i:i+1]

            rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)
            
            losses[f"L2_{fname}"] = rel_l2_val
            losses['mean_l2'] += rel_l2_val / num_channels
            losses[f"RMSE_{fname}"] = rmse[i].item()

    losses["each_l2"] = _each_l2(pred_real, label_real)

    return losses


def get_val_loss(fields, predict_hat, state, normalizer):
    num_channels = float(len(fields))

    pred_fp32 = predict_hat.detach().float()
    label_fp32 = state.detach().float()

    pred_real = normalizer.denormalize(pred_fp32)
    state_real = normalizer.denormalize(label_fp32)

    losses = {
        'mean_l2': 0
        }
    
    rmse = _rmse(pred_real, state_real)

    for i, fname in enumerate(fields):
        pred_ch_real = pred_real[..., i:i+1]
        gt_ch_real = state_real[..., i:i+1]
        
        rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)
        
        losses[f"L2_{fname}"] = rel_l2_val
        losses['mean_l2'] += rel_l2_val / num_channels
        losses[f"RMSE_{fname}"] = rmse[i].item()

    losses["each_l2"] = _each_l2(pred_real, state_real)
    
    return losses


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

        batch_num = state.shape[0]

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):   
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss)
            
            costs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()
                
        else:
            predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

            costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss)
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
    fields = args.data.get("fields", ["T"])
    use_amp, check_point = args.model.get("use_amp", False), args.model.get("check_point", False)
    agg = {}
    for key in ["L2", "mean_l2", "RMSE"]:
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
            edges = batch["edges"].to(device)
            time_seq = batch["time_seq"].to(device)
            conditions = batch["conditions"].to(device).float()

            batch_num = state.shape[0]

            if use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                    costs = get_val_loss(fields, predict_hat, state[:, 1:], normalizer)
            else:
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                costs = get_val_loss(fields, predict_hat, state[:, 1:], normalizer)

            for fname in fields:
                agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
                agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
            agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
            agg["each_l2"] += costs["each_l2"] * batch_num
            agg["num"] += batch_num
            # 只保存 batch 中的第一个（任意一个）样本
            # if i in viz_batch_indices:
            #     # 随机选择一个样本进行可视化
            #     sample_idx = random.randint(0, batch_num - 1)
            #     pred_real = normalizer.denormalize(predict_hat)
            #     gt_real = normalizer.denormalize(state[:, 1:])

            #     from src.utils import save_vtk_result
            #     save_vtk_result(save_dir=args.save_path, epoch=epoch, file_id=f"batch{i}_sample{sample_idx}",
            #                     predictions=pred_real[sample_idx], ground_truths=gt_real[sample_idx], node_pos=node_pos[0], field_names=fields)


    for key, value in agg.items():
        if key != "each_l2" and key != "num":
            agg[key] = value / agg["num"]
    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    return agg
