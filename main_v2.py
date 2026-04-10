"""
main_v2.py — 改进训练流程
============================================================
相比 main.py 的改进：
1. Warmup + CosineAnnealingWarmRestarts 调度器
   - 前 warmup_epochs 线性增长 LR，避免初期震荡
   - 多周期余弦退火 (T_0, T_mult) 帮助逃出局部最优
2. 更合理的 weight_decay (1e-4 vs 1e-5)
3. 可配置的 gradient loss 权重 (默认 5.0 vs 硬编码 8.0)
4. Pushforward training: 训练中后期逐步增加 rollout 步数
   - 缓解 autoregressive error accumulation
5. EMA (Exponential Moving Average) 模型，验证时使用 EMA 权重
6. 更频繁的 eval (每 10 epoch)
7. 支持从上一个 best checkpoint 继续训练
============================================================
"""

import torch
import numpy as np
import os
import time
import copy
import math
import traceback
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from src.dataset_fast import AeroGtoDataset
from src.dataset_2d import AeroGtoDataset2D
from src.dataset_cut_fast import CutAeroGtoDataset
from src.train import train, validate, get_train_loss, _init_region_agg, _accumulate_region, _finalize_region
from src.utils import set_seed, init_weights, parse_args, load_json_config


# =============================================================================
# Error logging helpers
# =============================================================================

def _write_error(path_record, args_name, exc, context="", config_path=""):
    """Write exception info to both training_log.txt and a dedicated bug.txt."""
    timestamp = time.asctime(time.localtime(time.time()))
    tb_str = traceback.format_exc()

    # ---- training_log.txt (brief) ----
    try:
        log_path = f"{path_record}/{args_name}_training_log.txt"
        with open(log_path, "a") as f:
            f.write(f"\n{'='*20} ERROR {'='*20}\n")
            f.write(f"Time: {timestamp}\n")
            if context:
                f.write(f"Context: {context}\n")
            f.write(f"{tb_str}\n")
    except Exception:
        pass  # avoid recursive error

    # ---- bug.txt (detailed) ----
    try:
        bug_path = f"{path_record}/bug.txt"
        with open(bug_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Time      : {timestamp}\n")
            f.write(f"Config    : {config_path}\n")
            f.write(f"Run name  : {args_name}\n")
            if context:
                f.write(f"Context   : {context}\n")
            f.write(f"Exception : {type(exc).__name__}: {exc}\n")
            f.write(f"Traceback :\n{tb_str}\n")
    except Exception:
        pass


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMA:
    """Maintains an exponential moving average of model parameters.

    IMPORTANT: call update() after every optimizer step (every batch),
    not once per epoch.  With decay=0.999 the effective averaging window
    is ~1000 steps.  If called only once per epoch (e.g. 192 batches/epoch),
    after 100 epochs the shadow still retains >90% of the *initial* weights,
    producing catastrophically bad inference.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 原位更新，避免每次 update 都 clone 一份新张量
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """Replace model params with EMA shadow (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].clone()

    def restore(self, model):
        """Restore original model params after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Warmup + CosineRestart Scheduler
# =============================================================================

class WarmupCosineScheduler:
    """
    Linear warmup followed by CosineAnnealingLR (monotonic decay).

    Previous WarmupCosineRestartScheduler used CosineAnnealingWarmRestarts which
    caused lr to bounce back to near-initial values late in training (e.g. epoch
    160+/200 with T_0=50, T_mult=2), preventing fine-grained convergence and
    causing severe overfitting (test L2 14x worse than main.py baseline).

    This version uses a single cosine decay after warmup, ensuring lr decreases
    monotonically to eta_min by the end of training.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.eta_min = eta_min
        self.current_epoch = 0

        # Cosine scheduler covers the post-warmup portion
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min
        )

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            self.cosine_scheduler.step()

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "cosine_scheduler": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state):
        self.current_epoch = state["current_epoch"]
        self.cosine_scheduler.load_state_dict(state["cosine_scheduler"])


# =============================================================================
# Pushforward Training
# =============================================================================

def train_pushforward(args, model, train_dataloader, optim, device, normalizer, extra_steps, ema=None):
    """
    Pushforward training: extend rollout beyond horizon_train by extra_steps.
    Only backpropagate through the extra steps (the model is already good at the original horizon).
    This forces the model to learn to correct its own errors.
    """
    from torch.amp import GradScaler, autocast

    base_horizon = args.data.get("horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp = args.train.get("use_amp", False)
    check_point = args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})
    grad_loss_weight = args.train.get("grad_loss_weight", 8.0)
    model_name = args.model.get("name", "PhysGTO")

    agg = {}
    for key in ["loss", "L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(base_horizon, device=device)
    agg["num"] = 0
    agg["value_loss"] = 0.0
    agg["grad_loss"] = 0.0
    has_region = False

    model.train()
    normalizer.to(device)
    scaler = GradScaler('cuda') if use_amp else None

    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3")

    pbar = tqdm(train_dataloader, desc="  Train(PF)", unit="bt", leave=True, ncols=120, colour='cyan')

    for batch in pbar:
        dt = batch['dt'].to(device)
        state_cpu = batch["state"]           # [B, T+1, N, C] — stay on CPU until T_pf known
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq_cpu = batch["time_seq"]     # [B, T, 1] — stay on CPU until T_pf known
        if _use_spatial:
            spatial_inform = batch["spatial_inform"].to(device)
        conditions = batch["conditions"].to(device).float()
        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch['grid_shape'].numpy()

        active_mask = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask[:, 1:].to(device)
            if not has_region:
                _init_region_agg(agg, fields)
                has_region = True

        batch_num = state_cpu.shape[0]
        T_total = time_seq_cpu.shape[1]
        T_pf = min(base_horizon + extra_steps, T_total)

        # Transfer only needed slices to device
        state = state_cpu[:, :T_pf + 1].to(device)
        time_seq = time_seq_cpu[:, :T_pf].to(device)

        # Slice active_mask for base horizon
        base_mask = active_mask[:, :base_horizon] if active_mask is not None else None

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if _use_spatial:
                    predict_hat = model.autoregressive(
                        state[:, 0], node_pos, edges, time_seq[:, :T_pf], spatial_inform, conditions, dt, check_point)
                else:
                    predict_hat = model.autoregressive(
                        state[:, 0], node_pos, edges, time_seq[:, :T_pf], conditions, dt, check_point)
                # Loss on original horizon
                costs = get_train_loss(fields, predict_hat[:, :base_horizon], state[:, 1:base_horizon+1], normalizer, weight_loss, active_mask=base_mask)
                loss_base = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

                # Loss on extra steps (pushforward)
                if T_pf > base_horizon and state.shape[1] > base_horizon + 1:
                    T_extra = min(T_pf, state.shape[1] - 1)
                    pf_mask = active_mask[:, base_horizon:T_extra] if active_mask is not None else None
                    costs_pf = get_train_loss(
                        fields, predict_hat[:, base_horizon:T_extra],
                        state[:, base_horizon+1:T_extra+1], normalizer, weight_loss, active_mask=pf_mask
                    )
                    loss_pf = costs_pf["value_loss"] + grad_loss_weight * costs_pf["grad_loss"]
                    total_loss = loss_base + 0.5 * loss_pf  # pushforward weighted less
                else:
                    total_loss = loss_base

            # NaN guard / Loss spike guard: 跳过异常 batch，并释放计算图显存
            _skip = False
            if not torch.isfinite(total_loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and total_loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss_base, total_loss
                if 'costs_pf' in locals(): del costs_pf, loss_pf
                torch.cuda.empty_cache()
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)
        else:
            if _use_spatial:
                predict_hat = model.autoregressive(
                    state[:, 0], node_pos, edges, time_seq[:, :T_pf], spatial_inform, conditions, dt, check_point
                )
            else:
                predict_hat = model.autoregressive(
                    state[:, 0], node_pos, edges, time_seq[:, :T_pf], conditions, dt, check_point
                )
            costs = get_train_loss(fields, predict_hat[:, :base_horizon], state[:, 1:base_horizon+1], normalizer, weight_loss, active_mask=base_mask)
            loss_base = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

            if T_pf > base_horizon and state.shape[1] > base_horizon + 1:
                T_extra = min(T_pf, state.shape[1] - 1)
                pf_mask = active_mask[:, base_horizon:T_extra] if active_mask is not None else None
                costs_pf = get_train_loss(
                    fields, predict_hat[:, base_horizon:T_extra],
                    state[:, base_horizon+1:T_extra+1], normalizer, weight_loss, active_mask=pf_mask
                )
                loss_pf = costs_pf["value_loss"] + grad_loss_weight * costs_pf["grad_loss"]
                total_loss = loss_base + 0.5 * loss_pf
            else:
                total_loss = loss_base

            # NaN guard / Loss spike guard
            _skip = False
            if not torch.isfinite(total_loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and total_loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss_base, total_loss
                if 'costs_pf' in locals(): del costs_pf, loss_pf
                torch.cuda.empty_cache()
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)

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
        pbar.set_postfix({"Loss": f"{avg_loss:.4e}"})

    from src.train import _REGION_PREFIXES, _REGION_MEANS
    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in ("active_loss", "inactive_loss") and key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=True)
    return agg


# =============================================================================
# Patched train function with configurable grad_loss_weight
# =============================================================================

def train_v2(args, model, train_dataloader, optim, device, normalizer, ema=None):
    """train() with configurable grad_loss_weight instead of hardcoded 8.0"""
    from torch.amp import GradScaler, autocast

    horizon = args.data.get("horizon_train", 1) if isinstance(args.data, dict) else getattr(args, "horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp = args.train.get("use_amp", False)
    check_point = args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})
    grad_loss_weight = args.train.get("grad_loss_weight", 8.0)
    model_name = args.model.get("name", "PhysGTO")

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
    has_region = False

    model.train()
    normalizer.to(device)
    scaler = GradScaler('cuda') if use_amp else None

    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3")

    pbar = tqdm(train_dataloader, desc="  Train", unit="bt", leave=True, ncols=120, colour='green')
    for batch in pbar:
        dt = batch['dt'].to(device)
        state = batch["state"][:, :horizon + 1].to(device)
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq = batch["time_seq"][:, :horizon].to(device)
        if _use_spatial:
            spatial_inform = batch["spatial_inform"].to(device)
        conditions = batch["conditions"].to(device).float()
        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch['grid_shape'].numpy()

        active_mask = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask[:, 1:horizon + 1].to(device)
            if not has_region:
                _init_region_agg(agg, fields)
                has_region = True

        batch_num = state.shape[0]

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if _use_spatial:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
                else:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

                costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)

            # Use configurable grad_loss_weight
            loss = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

            # NaN guard / Loss spike guard: 跳过异常 batch，并释放计算图显存
            _skip = False
            if not torch.isfinite(loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)
        else:
            if _use_spatial:
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
            else:
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

            costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)

            loss = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

            # NaN guard / Loss spike guard
            _skip = False
            if not torch.isfinite(loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)

        # For logging, store the combined loss into costs
        costs["loss"] = loss

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
        pbar.set_postfix({"Loss": f"{avg_loss:.4e}"})

    from src.train import _REGION_PREFIXES, _REGION_MEANS
    if agg["num"] == 0:
        raise RuntimeError(
            "train_v2: all batches were skipped (NaN/spike guard triggered every batch). "
            "Loss is NaN or Inf — check for gradient explosion or bad input data."
        )
    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in ("active_loss", "inactive_loss") and key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=True)
    return agg


# =============================================================================
# DataLoader (same as main.py)
# =============================================================================

def _worker_init_fn(worker_id, base_seed):
    """固定每个 DataLoader worker 的随机种子，确保 shuffle 顺序可复现。"""
    import random as _random
    seed = base_seed + worker_id
    np.random.seed(seed)
    _random.seed(seed)
    torch.manual_seed(seed)


def get_dataloader(args, path_record, device_type, pf_extra_max=0):
    data_cfg = args.data
    model_cfg = args.model
    space_dim = model_cfg.get("space_size", 3)

    if space_dim == 3:
        if data_cfg.get("cut", False):
            Datasetclass = CutAeroGtoDataset
        else:
            Datasetclass = AeroGtoDataset
    elif space_dim == 2:
        Datasetclass = AeroGtoDataset2D

    # Inject pf_extra_max into data_cfg so train dataset can load extra time steps
    if pf_extra_max > 0:
        data_cfg["horizon_pf_extra"] = pf_extra_max

    train_dataset = Datasetclass(
        args=args,
        mode="train"
    )

    test_dataset = Datasetclass(
        args=args,
        mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None
    )

    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()  # 同步 norm_mean/norm_std 缓存

    # Use 1/4 of the test set to reduce evaluation time
    subset_size = max(1, len(test_dataset) // 4)
    indices = list(range(0, len(test_dataset), 4))[:subset_size]
    test_dataset = Subset(test_dataset, indices)

    pin_memory = True if "cuda" in device_type else False

    seed = getattr(args, "seed", None)
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        dl_kwargs = dict(
            generator=g,
            worker_init_fn=lambda wid: _worker_init_fn(wid, seed),
        )
    else:
        dl_kwargs = {}

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg['train'].get("batchsize", 1),
        shuffle=True,
        num_workers=data_cfg['train'].get("num_workers", 0),
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_cfg['test'].get("batchsize", 1),
        shuffle=False,
        num_workers=data_cfg['test'].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    cond_dim = args.model.get("cond_dim") or train_dataset.cond_dim
    edge_num = train_dataset.meta_cache[train_dataset.file_paths[0]]["edges"].shape[0]

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
        file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
        file.write(f"Node num: {train_dataset.node_num}, Edge num: {edge_num}, Cond dim: {cond_dim}\n")
        file.write(f"Mean dt: {train_dataset.dt:.4e}\n")

    return train_dataloader, test_dataloader, train_dataset.normalizer, cond_dim, train_dataset.dt


def get_model(args, device, cond_dim, default_dt):
    model_cfg = args.model
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

    common_kwargs = dict(
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
        common_kwargs["n_fields"] = model_cfg.get("n_fields", model_cfg.get("in_dim", 2))
        common_kwargs["cross_attn_heads"] = model_cfg.get("cross_attn_heads", 4)

    if model_name in ("gto_attnres_multi_v2", "gto_res_attnres"):
        common_kwargs["attn_res_mode"] = model_cfg.get("attn_res_mode", "block_inter")

    if model_name in ("PhysGTO_v2", "gto_attnres_multi_v3"):
        common_kwargs["spatial_dim"] = model_cfg.get("spatial_dim", 10)
        common_kwargs["pos_x_boost"] = model_cfg.get("pos_x_boost", 2)

    if model_name == "gto_attnres_multi_v3":
        common_kwargs["n_latent"] = model_cfg.get("n_latent", 4)

    model = Model(**common_kwargs).to(device)

    load_path = model_cfg.get("load_path")
    checkpoint = None

    if load_path:
        model_path = os.path.join(load_path, f"{args.name}_best.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {model_path}")
    elif model_cfg.get("if_init", True):
        model.apply(init_weights)

    return model, checkpoint


# =============================================================================
# Main training loop
# =============================================================================

def main(args, path_logs, path_nn, path_record, config_path=""):

    device_str = args.device
    if "cuda" in device_str and not torch.cuda.is_available():
        print("! Warning: CUDA not available, using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    EPOCH = int(args.train["epoch"])
    real_lr = float(args.train["lr"])
    fields = args.data.get("fields", ["T"])

    # ---- Pushforward config (parsed early to size the dataset) ----
    pf_cfg = args.train.get("pushforward", {"enable": False})
    pf_enable = pf_cfg.get("enable", False)
    pf_start = pf_cfg.get("start_epoch", 80)
    pf_extra_max = pf_cfg.get("extra_steps", 3)
    pf_ramp = pf_cfg.get("ramp_epochs", 40)

    # Dataloader & normalizer
    train_dataloader, test_dataloader, normalizer, cond_dim, default_dt = get_dataloader(
        args, path_record, device_str,
        pf_extra_max=pf_extra_max if pf_enable else 0
    )

    # Model
    model, checkpoint = get_model(args, device, cond_dim, default_dt)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = int(sum([np.prod(p.size()) for p in model_parameters]))

    print(f"EPOCH: {EPOCH}, #params: {params/1e6:.2f}M")

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"Using device: {device}\n")
        file.write(f"{args.name}, #params: {params/1e6:.2f}M\n")
        file.write(f"EPOCH: {EPOCH}\n")
        file.write(f"Fields: {fields}\n")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{path_logs}/{args.name}_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # ---- Optimizer ----
    weight_decay = args.train.get("weight_decay", real_lr / 50.0)
    optimizer = AdamW(model.parameters(), lr=real_lr, weight_decay=weight_decay)

    # ---- Scheduler ----
    sched_cfg = args.train.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")

    if sched_type == "cosine_warmrestart":
        warmup_epochs = sched_cfg.get("warmup_epochs", 10)
        eta_min_ratio = sched_cfg.get("eta_min_ratio", 0.05)
        eta_min = real_lr * eta_min_ratio
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup_epochs,
            total_epochs=EPOCH, eta_min=eta_min
        )
        print(f"Scheduler: WarmupCosine (warmup={warmup_epochs}, total={EPOCH}, eta_min={eta_min:.2e})")
    else:
        # Fallback: original behavior
        if EPOCH < 50:
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=real_lr)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=real_lr / 20.0)
        print(f"Scheduler: CosineAnnealingLR (T_max={EPOCH})")

    # ---- EMA ----
    ema = EMA(model, decay=0.998)
    print("EMA enabled (decay=0.998)")

    if pf_enable:
        print(f"Pushforward: ON (start={pf_start}, extra_max={pf_extra_max}, ramp={pf_ramp})")

    # ---- Resume ----
    start_epoch, best_val_error = 0, float("inf")

    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Restored optimizer state.")
        if 'scheduler' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("Restored scheduler state.")
            except Exception:
                print("Scheduler state incompatible, starting fresh scheduler.")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        if 'best_val_error' in checkpoint:
            best_val_error = checkpoint['best_val_error']
            print(f"Restored best val error: {best_val_error:.4e}")

    if start_epoch >= EPOCH:
        print(f"Warning: Start epoch {start_epoch} >= Total EPOCH {EPOCH}. Training may perform 0 steps.")

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"Optimizer: AdamW (lr={real_lr}, wd={weight_decay})\n")
        file.write(f"Scheduler: {sched_type}\n")
        file.write(f"EMA: decay=0.998\n")
        file.write(f"Pushforward: enable={pf_enable}\n")
        file.write(f"grad_loss_weight: {args.train.get('grad_loss_weight', 8.0)}\n")

    # ---- Early-stop on repeated NaN/skip failures ----
    consecutive_skip_errors = 0
    max_consecutive_skip_errors = args.train.get("max_consecutive_errors", 3)

    # ---- Warn if dt is dangerously small for euler + AMP ----
    model_cfg = args.model
    train_cfg = args.train
    if (model_cfg.get("name") == "PhysGTO_v2"
            and model_cfg.get("stepper_scheme", "euler") == "euler"
            and train_cfg.get("use_amp", False)
            and not args.data.get("dt_scale", False)):
        _warn_msg = (
            "[WARNING] Potential NaN risk: PhysGTO_v2 with euler stepper + use_amp=True "
            "but dt_scale=False. Raw physical dt (e.g. 5e-6) forces v_pred to be ~1/dt "
            "larger, which overflows bfloat16. Consider setting dt_scale=true or use_amp=false."
        )
        print(_warn_msg)
        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(_warn_msg + "\n")

    # ==================== Training Loop ====================
    for epoch in range(start_epoch, EPOCH):
        start_time = time.time()

        # Determine if pushforward is active
        use_pushforward = False
        pf_extra = 0
        if pf_enable and epoch >= pf_start:
            progress = min(1.0, (epoch - pf_start) / max(1, pf_ramp))
            pf_extra = max(1, int(round(progress * pf_extra_max)))
            use_pushforward = True

        # Train
        try:
            if use_pushforward:
                train_error = train_pushforward(
                    args, model, train_dataloader, optimizer, device, normalizer,
                    extra_steps=pf_extra, ema=ema
                )
            else:
                train_error = train_v2(
                    args, model, train_dataloader, optimizer, device, normalizer,
                    ema=ema
                )
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            _write_error(path_record, args.name, e,
                         context=f"train epoch {epoch + 1}/{EPOCH} — CUDA OOM (fatal, stopping)",
                         config_path=config_path)
            print(f"[FATAL OOM] Epoch {epoch + 1}: CUDA out of memory. Training stopped.")
            print(f"Error details saved to: {path_record}/{args.name}_training_log.txt")
            print(f"Full bug report saved to: {path_record}/bug.txt")
            writer.close()
            return
        except RuntimeError as e:
            _write_error(path_record, args.name, e,
                         context=f"train epoch {epoch + 1}/{EPOCH}",
                         config_path=config_path)
            print(f"[ERROR] Epoch {epoch + 1} training failed: {e}. Skipping epoch.")
            scheduler.step()
            # Check if this is a repeated all-batches-skipped error
            if "all batches were skipped" in str(e):
                consecutive_skip_errors += 1
                if consecutive_skip_errors >= max_consecutive_skip_errors:
                    _fatal_msg = (
                        f"[FATAL] {consecutive_skip_errors} consecutive epochs with all batches skipped "
                        f"(NaN/Inf). Training aborted at epoch {epoch + 1}/{EPOCH}. "
                        f"Check for numerical instability (e.g. dt_scale, use_amp, lr)."
                    )
                    print(_fatal_msg)
                    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                        file.write(_fatal_msg + "\n")
                    writer.close()
                    return
            else:
                consecutive_skip_errors = 0
            continue
        except Exception as e:
            _write_error(path_record, args.name, e,
                         context=f"train epoch {epoch + 1}/{EPOCH}",
                         config_path=config_path)
            print(f"[ERROR] Epoch {epoch + 1} training failed: {e}. Skipping epoch.")
            scheduler.step()
            consecutive_skip_errors = 0
            continue

        consecutive_skip_errors = 0

        end_time = time.time()

        # EMA is now updated per-batch inside train_v2/train_pushforward

        # Step scheduler
        scheduler.step()
        if hasattr(scheduler, 'get_last_lr'):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        training_time = (end_time - start_time)

        # Extract metrics
        train_loss = train_error['loss']
        train_mean_l2 = train_error['mean_l2']
        each_t_l2 = train_error['each_l2']

        # Log
        log_str = f"Training, Epoch: {epoch + 1}/{EPOCH}, train Loss: {train_loss:.4e}, mean_l2: {train_mean_l2:.4e}"
        writer.add_scalar('lr/lr', current_lr, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('L2/train_mean_l2', train_mean_l2, epoch)

        l2_details = []
        rmse_details = []
        region_l2_details = []
        for fname in fields:
            l2_val = train_error[f"L2_{fname}"]
            rmse_val = train_error[f"RMSE_{fname}"]
            l2_details.append(f"{fname}: {l2_val:.4e}")
            rmse_details.append(f"{fname}: {rmse_val:.4e}")
            writer.add_scalar(f'L2/train_L2_{fname}', l2_val, epoch)
            writer.add_scalar(f'RMSE/train_RMSE_{fname}', rmse_val, epoch)

            for prefix in ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE"):
                rval = train_error.get(f"{prefix}_{fname}")
                if rval is not None and not math.isnan(rval):
                    writer.add_scalar(f'{prefix}/train_{prefix}_{fname}', rval, epoch)

            a_l2 = train_error.get(f"active_L2_{fname}")
            i_l2 = train_error.get(f"inactive_L2_{fname}")
            if a_l2 is not None:
                region_l2_details.append(f"{fname}: act={a_l2:.4e}, inact={i_l2:.4e}")

        for key in ("active_mean_l2", "inactive_mean_l2"):
            val = train_error.get(key)
            if val is not None and not math.isnan(val):
                writer.add_scalar(f'L2/train_{key}', val, epoch)
        for key in ("active_loss", "inactive_loss"):
            val = train_error.get(key)
            if val is not None:
                writer.add_scalar(f'Loss/train_{key}', val, epoch)

        print(log_str)
        value_loss = train_error.get("value_loss", 0)
        grad_loss = train_error.get("grad_loss", 0)
        print(f"value_loss:{value_loss} | grad_loss:{grad_loss}")
        print(f"L2 details: {', '.join(l2_details)}")
        print(f"RMSE details: {', '.join(rmse_details)}")
        if region_l2_details:
            print(f"Region L2: {', '.join(region_l2_details)}")
        print(f"each time step loss: {each_t_l2.tolist()}")
        pf_info = f", pushforward extra={pf_extra}" if use_pushforward else ""
        print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}{pf_info}")
        print("--------------")

        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(f"Training, epoch: {epoch + 1}/{EPOCH}\n")
            file.write(f"Train Loss: {train_loss:.4e}, mean_l2: {train_mean_l2:.4e}\n")
            file.write(f"L2 details: {', '.join(l2_details)}\n")
            file.write(f"RMSE details: {', '.join(rmse_details)}\n")
            if region_l2_details:
                file.write(f"Region L2: {', '.join(region_l2_details)}\n")
            file.write(f"each time step loss: {each_t_l2.tolist()}\n")
            file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}{pf_info}\n")

        # Validation (using EMA model)
        eval_every = args.train.get("eval_every", 5)
        if (epoch + 1) % eval_every == 0 or epoch == 0 or (epoch + 1) == EPOCH:
            start_time = time.time()

            try:
                # Apply EMA weights for evaluation
                torch.cuda.empty_cache()
                ema.apply_shadow(model)
                test_error = validate(args, model, test_dataloader, device, normalizer, epoch + 1)
                ema.restore(model)
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                ema.restore(model)
                torch.cuda.empty_cache()
                _write_error(path_record, args.name, e,
                             context=f"validate epoch {epoch + 1}/{EPOCH} — CUDA OOM (fatal, stopping)",
                             config_path=config_path)
                print(f"[FATAL OOM] Epoch {epoch + 1} validation: CUDA out of memory. Training stopped.")
                print(f"Error details saved to: {path_record}/{args.name}_training_log.txt")
                print(f"Full bug report saved to: {path_record}/bug.txt")
                writer.close()
                return
            except Exception as e:
                ema.restore(model)
                torch.cuda.empty_cache()
                _write_error(path_record, args.name, e,
                             context=f"validate epoch {epoch + 1}/{EPOCH}",
                             config_path=config_path)
                print(f"[ERROR] Epoch {epoch + 1} validation failed: {e}. Skipping validation.")
                continue

            end_time = time.time()
            val_time = (end_time - start_time)

            test_mean_l2 = test_error['mean_l2']
            test_each_t_l2 = test_error['each_l2']

            test_l2_details = []
            test_rmse_details = []
            test_region_l2_details = []
            writer.add_scalar('L2/test_mean_l2', test_mean_l2, epoch)

            for fname in fields:
                l2_val = test_error[f"L2_{fname}"]
                rmse_val = test_error[f"RMSE_{fname}"]
                test_l2_details.append(f"{fname}: {l2_val:.4e}")
                test_rmse_details.append(f"{fname}: {rmse_val:.4e}")
                writer.add_scalar(f'L2/test_L2_{fname}', l2_val, epoch)
                writer.add_scalar(f'RMSE/test_RMSE_{fname}', rmse_val, epoch)

                for prefix in ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE"):
                    rval = test_error.get(f"{prefix}_{fname}")
                    if rval is not None and not math.isnan(rval):
                        writer.add_scalar(f'{prefix}/test_{prefix}_{fname}', rval, epoch)

                a_l2 = test_error.get(f"active_L2_{fname}")
                i_l2 = test_error.get(f"inactive_L2_{fname}")
                if a_l2 is not None:
                    test_region_l2_details.append(f"{fname}: act={a_l2:.4e}, inact={i_l2:.4e}")

            for key in ("active_mean_l2", "inactive_mean_l2"):
                val = test_error.get(key)
                if val is not None and not math.isnan(val):
                    writer.add_scalar(f'L2/test_{key}', val, epoch)

            print("---Inference (EMA)---")
            print(f"Epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}")
            print(f"L2 details: {', '.join(test_l2_details)}")
            print(f"RMSE details: {', '.join(test_rmse_details)}")
            if test_region_l2_details:
                print(f"Region L2: {', '.join(test_region_l2_details)}")
            print(f"each time step loss: {test_each_t_l2.tolist()}")
            print(f"time pre test epoch/s:{val_time:.2f}")
            print("--------------")

            with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                file.write(f"Inference(EMA), epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}\n")
                file.write(f"L2 details: {', '.join(test_l2_details)}\n")
                file.write(f"RMSE details: {', '.join(test_rmse_details)}\n")
                if test_region_l2_details:
                    file.write(f"Region L2: {', '.join(test_region_l2_details)}\n")
                file.write(f"each time step loss: {test_each_t_l2.tolist()}\n")
                file.write(f"time pre test epoch/s:{val_time:.2f}\n")

            # Save Best
            if args.if_save and test_mean_l2 < best_val_error:
                best_val_error = test_mean_l2
                ckpt = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_error': best_val_error,
                    'config': args
                }
                torch.save(ckpt, f"{path_nn}/{args.name}_best.pt")
                print(f"  >> New best! test_mean_l2={best_val_error:.4e}")

        # Regular Save
        if (epoch + 1) % 50 == 0 or (epoch + 1) == EPOCH:
            if args.if_save:
                ckpt = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'learning_rate': current_lr,
                    'best_val_error': best_val_error,
                }
                torch.save(ckpt, f"{path_nn}/{args.name}_{epoch+1}.pt")

    writer.close()


if __name__ == "__main__":
    cli_args = parse_args()
    args = load_json_config(cli_args.config)

    print(args)

    path_logs = args.save_path + "/logs"
    path_nn = args.save_path + "/nn"
    path_record = args.save_path + "/record"

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)
    os.makedirs(path_nn, exist_ok=True)
    os.makedirs(path_record, exist_ok=True)

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"{'='*20} Start (main_v2) {'='*20}\n")
        file.write(str(args) + "\n")
        file.write(f"Config file: {cli_args.config}\n")
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")

    if args.seed is not None:
        set_seed(args.seed)

    try:
        main(args, path_logs, path_nn, path_record, config_path=cli_args.config)
    except Exception as e:
        _write_error(path_record, args.name, e,
                     context="main training loop (fatal crash)",
                     config_path=cli_args.config)
        print(f"[FATAL ERROR] Training crashed: {e}")
        print(f"Error details saved to: {path_record}/{args.name}_training_log.txt")
        print(f"Full bug report saved to: {path_record}/bug.txt")
        raise
    finally:
        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
