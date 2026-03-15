"""
可替代 src/train.py 的训练模块（不改动原文件）。

新增接口（写在 config 的 train 节点）:
1) vf_loss: 针对 alpha/gamma 字段启用 BCE + Dice
2) vf_loss_weight: 针对体积分数字段做区间加权（默认 0.2~0.8）

示例:
"train": {
  ...
  "vf_loss": {
    "enabled": true,
    "target_fields": ["alpha.air", "gamma_liquid"],
    "bce_weight": 1.0,
    "dice_weight": 1.0,
    "from_logits": false,
    "eps": 1e-6
  },
  "vf_loss_weight": {
    "enabled": true,
    "target_fields": ["alpha.air", "gamma_liquid"],
    "type": "range",
    "low_val": 0.2,
    "high_val": 0.8,
    "high_weight": 5.0,
    "bg_weight": 1.0,
    "use_phys": true
  }
}
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _relative_l2(pred, target):
    """相对L2误差，返回 [batch] 张量。"""
    error = pred - target
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-6)
    norm_error_channel = torch.mean(norm_error, dim=-1)
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


def _rmse(pred, target):
    diff = pred - target
    mse = torch.mean(diff ** 2, dim=[0, 1, 2])
    return torch.sqrt(mse)


def _to_list(obj):
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _weighted_mean(x, w, eps=1e-6):
    return torch.sum(x * w) / (torch.sum(w) + eps)


def _weighted_dice_loss(prob, target, w, eps=1e-6):
    prob_f = prob.reshape(prob.shape[0], -1)
    target_f = target.reshape(target.shape[0], -1)
    w_f = w.reshape(w.shape[0], -1)

    inter = torch.sum(w_f * prob_f * target_f, dim=1)
    denom = torch.sum(w_f * prob_f, dim=1) + torch.sum(w_f * target_f, dim=1)

    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _get_field_error_map(
    pred_norm,
    gt_norm,
    pred_phys,
    gt_phys,
    field_name,
    field_idx,
    calc_mode,
):
    if calc_mode == "mse_norm":
        return (pred_norm[..., field_idx:field_idx + 1] - gt_norm[..., field_idx:field_idx + 1]) ** 2

    if calc_mode == "rel_phys":
        if field_name == "T":
            p = pred_phys[..., field_idx:field_idx + 1]
            g = gt_phys[..., field_idx:field_idx + 1]
            return ((p - g) / (torch.abs(g) + 1e-6)) ** 2
        return (pred_norm[..., field_idx:field_idx + 1] - gt_norm[..., field_idx:field_idx + 1]) ** 2

    raise ValueError(f"Unsupported calc_mode: {calc_mode}")


def _apply_weight_rule(channel_weights, source_for_mask, fields, rule):
    if not isinstance(rule, dict):
        return
    if not rule.get("enabled", False):
        return

    target_fields = _to_list(rule.get("target_fields", rule.get("target_field")))
    if len(target_fields) == 0:
        return

    mode = rule.get("type", "threshold")
    high_weight = float(rule.get("high_weight", 10.0))
    bg_weight = float(rule.get("bg_weight", 1.0))
    apply_to_all = bool(rule.get("apply_to_all_channels", False))

    for fname in target_fields:
        if fname not in fields:
            continue
        f_idx = fields.index(fname)
        gt_target = source_for_mask[..., f_idx:f_idx + 1]

        if mode in ("threshold", "adaptive_threshold"):
            threshold_val = float(rule.get("threshold_val", 0.5))
            mask = gt_target > threshold_val
        elif mode in ("range", "band", "interval"):
            low_val = float(rule.get("low_val", 0.2))
            high_val = float(rule.get("high_val", 0.8))
            mask = (gt_target >= low_val) & (gt_target <= high_val)
        else:
            continue

        w_map = torch.full_like(gt_target, bg_weight)
        w_map[mask] = high_weight

        if apply_to_all:
            channel_weights *= w_map
        else:
            channel_weights[..., f_idx:f_idx + 1] *= w_map


def _build_channel_weights(label_gt, label_phys, fields, train_args, sample_weights):
    channel_weights = torch.ones_like(label_gt)

    if sample_weights is not None:
        channel_weights = channel_weights * sample_weights

    # 兼容旧接口: train.loss_weight
    legacy_cfg = train_args.get("loss_weight", None)
    if isinstance(legacy_cfg, dict):
        legacy_rule = dict(legacy_cfg)
        legacy_rule["enabled"] = legacy_rule.get("enabled", False)
        legacy_rule["apply_to_all_channels"] = legacy_rule.get("apply_to_all_channels", True)
        legacy_rule["type"] = legacy_rule.get("type", "threshold")
        mask_source = label_phys if legacy_rule.get("use_phys", True) else label_gt
        _apply_weight_rule(channel_weights, mask_source, fields, legacy_rule)

    # 新接口: 体积分数字段加权
    vf_weight_cfg = train_args.get("vf_loss_weight", None)
    if isinstance(vf_weight_cfg, dict):
        vf_rule = dict(vf_weight_cfg)
        vf_rule["enabled"] = vf_rule.get("enabled", False)
        vf_rule["type"] = vf_rule.get("type", "range")
        vf_rule["target_fields"] = vf_rule.get("target_fields", ["alpha.air", "gamma_liquid"])
        vf_rule["low_val"] = vf_rule.get("low_val", 0.2)
        vf_rule["high_val"] = vf_rule.get("high_val", 0.8)
        vf_rule["apply_to_all_channels"] = vf_rule.get("apply_to_all_channels", False)
        mask_source = label_phys if vf_rule.get("use_phys", True) else label_gt
        _apply_weight_rule(channel_weights, mask_source, fields, vf_rule)

    # 可选的通用扩展接口
    extra_rules = train_args.get("field_weight_rules", [])
    if isinstance(extra_rules, list):
        for rule in extra_rules:
            if not isinstance(rule, dict):
                continue
            mask_source = label_phys if rule.get("use_phys", True) else label_gt
            _apply_weight_rule(channel_weights, mask_source, fields, rule)

    return channel_weights


def _get_vf_loss_fields(fields, train_args):
    vf_cfg = train_args.get("vf_loss", {})
    target_fields = _to_list(vf_cfg.get("target_fields", None))
    if len(target_fields) > 0:
        return [f for f in target_fields if f in fields]

    # 未显式配置时，默认匹配 alpha* / gamma*
    default_fields = []
    for fname in fields:
        if fname.startswith("alpha") or fname.startswith("gamma"):
            default_fields.append(fname)
    return default_fields


def get_hybrid_loss(predict_hat, label_gt, normalizer, fields, train_args, sample_weights=None):
    """
    混合损失:
    - 默认字段: MSE / rel-phys (与旧版一致)
    - 体积分数字段(alpha/gamma): BCE + Dice
    - 支持 train.loss_weight(旧接口) + train.vf_loss_weight(新接口)
    """
    calc_mode = train_args.get("calc_mode", "mse_norm")

    pred_norm, gt_norm = predict_hat, label_gt
    pred_phys, gt_phys = normalizer.denormalize(predict_hat), normalizer.denormalize(label_gt)

    channel_weights = _build_channel_weights(label_gt, gt_phys, fields, train_args, sample_weights)

    vf_cfg = train_args.get("vf_loss", {})
    vf_enabled = bool(vf_cfg.get("enabled", False))
    vf_fields = set(_get_vf_loss_fields(fields, train_args))
    bce_w = float(vf_cfg.get("bce_weight", 1.0))
    dice_w = float(vf_cfg.get("dice_weight", 1.0))
    from_logits = bool(vf_cfg.get("from_logits", False))
    eps = float(vf_cfg.get("eps", 1e-6))

    per_channel_losses = []

    for i, fname in enumerate(fields):
        w = channel_weights[..., i:i + 1]

        if vf_enabled and fname in vf_fields:
            gt_prob = gt_phys[..., i:i + 1].clamp(0.0, 1.0)

            if from_logits:
                logits = pred_phys[..., i:i + 1]
                bce_map = F.binary_cross_entropy_with_logits(logits, gt_prob, reduction="none")
                pred_prob = torch.sigmoid(logits)
            else:
                pred_prob = pred_phys[..., i:i + 1].clamp(eps, 1.0 - eps)
                bce_map = F.binary_cross_entropy(pred_prob, gt_prob, reduction="none")

            bce_loss = _weighted_mean(bce_map, w, eps=eps)
            dice_loss = _weighted_dice_loss(pred_prob, gt_prob, w, eps=eps)
            ch_loss = bce_w * bce_loss + dice_w * dice_loss
        else:
            err_map = _get_field_error_map(
                pred_norm=pred_norm,
                gt_norm=gt_norm,
                pred_phys=pred_phys,
                gt_phys=gt_phys,
                field_name=fname,
                field_idx=i,
                calc_mode=calc_mode,
            )
            ch_loss = _weighted_mean(err_map, w, eps=1e-6)

        per_channel_losses.append(ch_loss)

    return torch.mean(torch.stack(per_channel_losses))


def get_train_loss(args, predict_hat, label_gt, normalizer, weights=None):
    train_args = args.train
    fields = args.data.get("fields", ["T"])

    loss_val = get_hybrid_loss(
        predict_hat=predict_hat,
        label_gt=label_gt,
        normalizer=normalizer,
        fields=fields,
        train_args=train_args,
        sample_weights=None,
    )

    num_channels = float(len(fields))
    losses = {"loss": loss_val, "mean_l2": 0}

    with torch.no_grad():
        pred_real = normalizer.denormalize(predict_hat)
        label_real = normalizer.denormalize(label_gt)
        rmse = _rmse(pred_real, label_real)

        for i, fname in enumerate(fields):
            pred_ch_real = pred_real[..., i:i + 1]
            gt_ch_real = label_real[..., i:i + 1]

            rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)
            losses[f"L2_{fname}"] = rel_l2_val
            losses["mean_l2"] += rel_l2_val / num_channels
            losses[f"RMSE_{fname}"] = rmse[i].item()

    losses["each_l2"] = _each_l2(pred_real, label_real)
    return losses


def get_val_loss(fields, predict_hat, state, normalizer):
    num_channels = float(len(fields))

    pred_real = normalizer.denormalize(predict_hat)
    state_real = normalizer.denormalize(state)

    losses = {"mean_l2": 0}
    rmse = _rmse(pred_real, state_real)

    for i, fname in enumerate(fields):
        pred_ch_real = pred_real[..., i:i + 1]
        gt_ch_real = state_real[..., i:i + 1]

        rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)
        losses[f"L2_{fname}"] = rel_l2_val
        losses["mean_l2"] += rel_l2_val / num_channels
        losses[f"RMSE_{fname}"] = rmse[i].item()

    losses["each_l2"] = _each_l2(pred_real, state_real)
    return losses


def train(args, model, train_dataloader, optim, device, normalizer):
    horizon = args.data.get("horizon_train", 1) if isinstance(args.data, dict) else getattr(args, "horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    teacher = args.train.get("teacher", False)

    agg = {}
    for key in ["loss", "L2", "mean_l2", "RMSE"]:
        if key in ("L2", "RMSE"):
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0

    model.train()
    normalizer.to(device)

    pbar = tqdm(train_dataloader, desc="  Train", unit="bt", leave=True, ncols=120, colour="green")
    for batch in pbar:
        dt = batch["dt"].to(device)
        state = batch["state"].to(device)
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq = batch["time_seq"].to(device)
        conditions = batch["conditions"].to(device).float()
        raw_weights = batch["loss_weight"].to(device)
        step_weights = raw_weights[:, 1:, :, :]

        batch_num = state.shape[0]

        predict_hat = model.autoregressive(
            state[:, 0], node_pos, edges, time_seq, conditions, dt, teacher, state[:, 1:]
        )
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
        pbar.set_postfix({"Loss": f"{avg_loss:.4e}"})

    for key, value in agg.items():
        if key not in ("each_l2", "num"):
            agg[key] = value / agg["num"]
    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    return agg


def validate(args, model, val_dataloader, device, normalizer, epoch):
    horizon = args.data.get("horizon_test", 1) if isinstance(args.data, dict) else getattr(args, "horizon_test", 1)
    fields = args.data.get("fields", ["T"])

    agg = {}
    for key in ["L2", "mean_l2", "RMSE"]:
        if key in ("L2", "RMSE"):
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0

    model.eval()
    normalizer.to(device)

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="  Valid", unit="bt", leave=False, ncols=120, colour="yellow")
        for batch in pbar:
            dt = batch["dt"].to(device)
            state = batch["state"].to(device)
            node_pos = batch["node_pos"].to(device)
            edges = batch["edges"].to(device)
            time_seq = batch["time_seq"].to(device)
            conditions = batch["conditions"].to(device).float()

            batch_num = state.shape[0]
            predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt)
            costs = get_val_loss(fields, predict_hat, state[:, 1:], normalizer)

            for fname in fields:
                agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
                agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
            agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
            agg["each_l2"] += costs["each_l2"] * batch_num
            agg["num"] += batch_num

    for key, value in agg.items():
        if key not in ("each_l2", "num"):
            agg[key] = value / agg["num"]
    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    return agg
