import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from pathlib import Path

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

def get_weighted_mse_loss(predict_hat, label_gt, normalizer, fields, train_args, device):
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

        # gt_target_norm = label_gt[..., f_idx:f_idx+1]
        gt_target_phys = gt_denorm[..., f_idx:f_idx+1]

        threshold_val = weight_cfg.get("threshold_val", 500.0)
        mask = gt_target_phys > threshold_val
    
        weights[mask] = weight_cfg.get("high_weight", 10.0)
        
    weighted_sq_error = base_error_map * weights
    
    loss = torch.mean(weighted_sq_error)

    return loss

def get_train_loss(args, predict_hat, label_gt, normalizer, weights=None):
    """返回loss张量及监控指标（其余转为float）。"""
    train_args = args.train
    fields = args.data.get("fields", ["T"])
    loss_flag = train_args.get("loss_flag", "L2_norm_loss")

    loss_val = get_weighted_mse_loss(predict_hat, label_gt, normalizer, fields, train_args, device=predict_hat.device)

    num_channels = float(len(fields))

    # if loss_flag == "L2_norm_loss":
    #     base_pred, base_label = predict_hat, label_gt
    # else:
    #     base_pred = normalizer.denormalize(predict_hat)
    #     base_label = normalizer.denormalize(label_gt)

    losses = {
        "loss": loss_val,
        'mean_l2': 0
        }
    
    with torch.no_grad():
        pred_real = normalizer.denormalize(predict_hat)
        label_real = normalizer.denormalize(label_gt)
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

    pred_real = normalizer.denormalize(predict_hat)
    state_real = normalizer.denormalize(state)

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
    teacher = args.train.get("teacher", False) # 开启 teacher forcing
    agg = {}
    for key in ["loss", "L2", "mean_l2", "RMSE"]:
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
        edges = batch["edges"].to(device)
        time_seq = batch["time_seq"].to(device)
        conditions = batch["conditions"].to(device).float()
        raw_weights = batch["loss_weight"].to(device)
        step_weights = raw_weights[:, 1:, :, :] # [B, T, N, 1]

        batch_num = state.shape[0]
        
        predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, teacher, state[:, 1:])
        # predict_hat = torch.stack(pred_list, dim=1)

        costs = get_train_loss(args, predict_hat, state[:, 1:], normalizer, step_weights)

        optim.zero_grad(set_to_none=True)
        costs["loss"].backward()

        grad_clip = args.train.get("grad_clip", None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optim.step()


        agg["loss"] += costs["loss"].item() * batch_num
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
            predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt)
            # predict_hat = torch.stack(pred_list, dim=1)
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
