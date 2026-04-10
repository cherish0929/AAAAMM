"""
evaluate.py — 统一评估多个已训练模型在测试集上的表现
============================================================
用法：
  直接编辑 CONFIG_LIST，然后运行  python evaluate.py
  评估报告会同时打印到终端和保存到 result/evaluate_report_{timestamp}.txt
============================================================
"""

import torch
import numpy as np
import os
import sys
import math
import time
import traceback
from datetime import datetime
from types import SimpleNamespace

from src.utils import load_json_config, set_seed
from src.train import validate

# ============================================================
# >>> 在这里填写需要评估的 config 路径 <<<
# ============================================================
CONFIG_LIST = [
    # "config/easypool/GTO_easypool.json",
    # "config/easypool/GTO_easypool_stronger.json",
    # "config/easypool/GTO_attnres_easypool.json",
    # "config/easypool/GTO_attnres_easypool_stronger.json",
    # "config/easypool/GTO_2_easypool_stronger.json",
    # "config/easypool/GTO_attnres_3_easypool_stronger.json",
    # "config/easypool/cut_GTO_easypool.json",
    # "config/easypool/cut_GTO_attnres_easypool.json",
    # "config/easypool/cut_GTO_attnres_3_easypool.json",
    "config/keyhole/GTO_keyhole_stronger.json",
    "config/keyhole/GTO_attnres_keyhole_stronger.json",
    "config/keyhole/GTO_attnres_3_keyhole_stronger.json",
]


# ============================================================
# Logger: 同时 print 到终端 和 写入文件
# ============================================================
class DualLogger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.terminal = sys.stdout

    def log(self, msg=""):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# 从 main_v2.py 复用的 dataloader / model 加载逻辑
# ============================================================

def get_dataloader_eval(args, device_type):
    """与 main_v2.get_dataloader 相同, 但不写日志文件。"""
    from torch.utils.data import DataLoader, Subset
    from src.dataset_fast import AeroGtoDataset
    from src.dataset_2d import AeroGtoDataset2D
    from src.dataset_cut_fast import CutAeroGtoDataset

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

    train_dataset = Datasetclass(args=args, mode="train")

    test_dataset = Datasetclass(
        args=args, mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None,
    )
    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()

    # Use 1/4 of the test set to reduce evaluation time
    subset_size = max(1, len(test_dataset) // 4)
    indices = list(range(0, len(test_dataset), 4))[:subset_size]
    test_dataset = Subset(test_dataset, indices)

    pin_memory = "cuda" in device_type
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_cfg['test'].get("batchsize", 1),
        shuffle=False,
        num_workers=data_cfg['test'].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    cond_dim = args.model.get("cond_dim") or train_dataset.cond_dim
    default_dt = train_dataset.dt

    return test_dataloader, train_dataset.normalizer, cond_dim, default_dt


def load_model_and_checkpoint(args, device, cond_dim, default_dt):
    """
    实例化模型并加载 best checkpoint + EMA shadow。
    返回 (model, checkpoint_info_dict) 或在找不到 checkpoint 时返回 (None, error_msg)。
    """
    from src.utils import init_weights
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")

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

    # 加载 checkpoint
    path_nn = args.save_path + "/nn"
    ckpt_path = os.path.join(path_nn, f"{args.name}_best.pt")
    if not os.path.exists(ckpt_path):
        return None, f"Checkpoint not found: {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 优先使用 EMA shadow 权重
    if "ema_shadow" in checkpoint:
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = model.state_dict()
        for name in ema_shadow:
            if name in state_dict:
                state_dict[name] = ema_shadow[name]
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    ckpt_info = {
        "path": ckpt_path,
        "epoch": checkpoint.get("epoch", "N/A"),
        "best_val_error": checkpoint.get("best_val_error", "N/A"),
    }
    return model, ckpt_info


# ============================================================
# 打印 config 关键信息
# ============================================================

def print_config_info(logger, config_path, args):
    data_cfg = args.data
    model_cfg = args.model
    train_cfg = args.train

    logger.log(f"  Config file     : {config_path}")
    logger.log(f"  Experiment name : {args.name}")
    logger.log(f"  Save path       : {args.save_path}")

    # Model
    logger.log(f"  Model           : {model_cfg.get('name', 'PhysGTO')}")
    logger.log(f"  space_size={model_cfg.get('space_size', 3)}, "
               f"in_dim={model_cfg.get('in_dim', '?')}, out_dim={model_cfg.get('out_dim', '?')}, "
               f"enc_dim={model_cfg.get('enc_dim', 128)}, N_block={model_cfg.get('N_block', 4)}, "
               f"n_head={model_cfg.get('n_head', 4)}, n_token={model_cfg.get('n_token', 64)}")
    if model_cfg.get("n_fields"):
        logger.log(f"  n_fields={model_cfg['n_fields']}, cross_attn_heads={model_cfg.get('cross_attn_heads', 4)}")
    if model_cfg.get("attn_res_mode"):
        logger.log(f"  attn_res_mode   : {model_cfg['attn_res_mode']}")
    if model_cfg.get("stepper_scheme"):
        logger.log(f"  stepper_scheme  : {model_cfg['stepper_scheme']}")

    # Data
    logger.log(f"  Fields          : {data_cfg.get('fields', ['T'])}")
    logger.log(f"  Horizon (train) : {data_cfg.get('horizon_train', '?')}")
    logger.log(f"  Horizon (test)  : {data_cfg.get('horizon_test', '?')}")
    logger.log(f"  spatial_stride  : {data_cfg.get('spatial_stride', 'N/A')}")
    logger.log(f"  cut             : {data_cfg.get('cut', False)}")
    logger.log(f"  normalize       : {data_cfg.get('normalize', False)}")
    logger.log(f"  train_list      : {data_cfg.get('train_list', [])}")
    logger.log(f"  test_list       : {data_cfg.get('test_list', [])}")
    if data_cfg.get("active_mask"):
        logger.log(f"  active_mask     : {data_cfg['active_mask']}")

    # Train
    logger.log(f"  epoch           : {train_cfg.get('epoch', '?')}")
    logger.log(f"  lr              : {train_cfg.get('lr', '?')}")
    logger.log(f"  weight_decay    : {train_cfg.get('weight_decay', 'default')}")
    logger.log(f"  grad_loss_weight: {train_cfg.get('grad_loss_weight', 8.0)}")
    logger.log(f"  grad_clip       : {train_cfg.get('grad_clip', 'N/A')}")
    logger.log(f"  use_amp         : {train_cfg.get('use_amp', False)}")
    logger.log(f"  check_point     : {train_cfg.get('check_point', False)}")

    sched = train_cfg.get("scheduler", {})
    logger.log(f"  scheduler       : {sched.get('type', 'cosine')} "
               f"(warmup={sched.get('warmup_epochs', 'N/A')}, eta_min_ratio={sched.get('eta_min_ratio', 'N/A')})")

    wl = train_cfg.get("weight_loss", {})
    if wl.get("enable", False):
        logger.log(f"  weight_loss     : field={wl.get('field')}, thresh={wl.get('threshold')}, "
                   f"base_w={wl.get('base_weight')}, focus_w={wl.get('focus_weight')}")
        if wl.get("gradient", False):
            logger.log(f"  gradient_loss   : grad_weight={wl.get('grad_weight')}")
    else:
        logger.log(f"  weight_loss     : disabled")

    pf = train_cfg.get("pushforward", {})
    if pf.get("enable", False):
        logger.log(f"  pushforward     : start={pf.get('start_epoch')}, extra={pf.get('extra_steps')}, ramp={pf.get('ramp_epochs')}")
    else:
        logger.log(f"  pushforward     : disabled")


# ============================================================
# 打印评估结果
# ============================================================

def print_metrics(logger, metrics, fields):
    logger.log(f"  mean_l2         : {metrics['mean_l2']:.6e}")

    # Per-field L2 and RMSE
    for fname in fields:
        l2_val = metrics.get(f"L2_{fname}", "N/A")
        rmse_val = metrics.get(f"RMSE_{fname}", "N/A")
        line = f"  {fname:16s}: L2={l2_val:.6e}, RMSE={rmse_val:.6e}" if isinstance(l2_val, float) else f"  {fname:16s}: L2={l2_val}, RMSE={rmse_val}"
        logger.log(line)

    # Region metrics
    has_region = any(k.startswith("active_L2_") for k in metrics)
    if has_region:
        logger.log("  --- Region Metrics ---")
        a_mean = metrics.get("active_mean_l2", float('nan'))
        i_mean = metrics.get("inactive_mean_l2", float('nan'))
        if not math.isnan(a_mean):
            logger.log(f"  active_mean_l2  : {a_mean:.6e}")
        if not math.isnan(i_mean):
            logger.log(f"  inactive_mean_l2: {i_mean:.6e}")

        for fname in fields:
            parts = []
            for prefix in ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE"):
                val = metrics.get(f"{prefix}_{fname}")
                if val is not None and not math.isnan(val):
                    parts.append(f"{prefix}={val:.6e}")
            if parts:
                logger.log(f"  {fname:16s}: {', '.join(parts)}")

    # Per-timestep L2
    each_l2 = metrics.get("each_l2")
    if each_l2 is not None:
        if isinstance(each_l2, torch.Tensor):
            each_l2 = each_l2.tolist()
        formatted = [f"{v:.4e}" for v in each_l2]
        logger.log(f"  each_step_l2    : [{', '.join(formatted)}]")


# ============================================================
# 评估单个 config
# ============================================================

def evaluate_single(config_path, device, logger):
    """
    评估单个配置文件对应的模型。
    返回 (args.name, fields, metrics_dict) 或 None（失败时）。
    """
    args = load_json_config(config_path)
    args.device = str(device)
    fields = args.data.get("fields", ["T"])

    if args.seed is not None:
        set_seed(args.seed)

    logger.log(f"\n{'='*70}")
    logger.log(f"Evaluating: {args.name}")
    logger.log(f"{'='*70}")

    print_config_info(logger, config_path, args)
    logger.log(f"  {'- '*35}")

    # 加载数据
    logger.log("  Loading dataset ...")
    test_dataloader, normalizer, cond_dim, default_dt = get_dataloader_eval(args, str(device))
    logger.log(f"  Test samples: {len(test_dataloader.dataset)}, Test batches: {len(test_dataloader)}")

    # 加载模型 + checkpoint
    logger.log("  Loading model + checkpoint ...")
    model, ckpt_info = load_model_and_checkpoint(args, device, cond_dim, default_dt)
    if model is None:
        logger.log(f"  [SKIP] {ckpt_info}")
        return None

    logger.log(f"  Checkpoint      : {ckpt_info['path']}")
    logger.log(f"  Trained epoch   : {ckpt_info['epoch']}")
    logger.log(f"  best_val_error  : {ckpt_info['best_val_error']}")

    # 参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  #params         : {params/1e6:.2f}M")

    # 评估
    logger.log("  Running validation ...")
    t0 = time.time()
    model.eval()
    normalizer.to(device)
    metrics = validate(args, model, test_dataloader, device, normalizer, epoch=0)
    elapsed = time.time() - t0
    logger.log(f"  Validation time : {elapsed:.1f}s")

    logger.log(f"  --- Results ---")
    print_metrics(logger, metrics, fields)

    metrics["_best_val_error"] = ckpt_info["best_val_error"]
    return args.name, fields, metrics


# ============================================================
# 汇总表格
# ============================================================

def print_summary_table(logger, results):
    """打印一个简洁的横向对比汇总表。"""
    if not results:
        logger.log("\nNo successful evaluations to summarize.")
        return

    logger.log(f"\n{'='*70}")
    logger.log("SUMMARY TABLE")
    logger.log(f"{'='*70}")

    # 收集所有出现过的字段名
    all_fields = []
    for name, fields, metrics in results:
        for f in fields:
            if f not in all_fields:
                all_fields.append(f)

    # 表头
    header_parts = [f"{'Name':40s}", f"{'mean_l2':>12s}"]
    for f in all_fields:
        header_parts.append(f"{'L2_'+f:>14s}")
        header_parts.append(f"{'RMSE_'+f:>14s}")
    # Region columns
    header_parts.append(f"{'act_mean_l2':>14s}")
    header_parts.append(f"{'inact_mean_l2':>14s}")
    header_parts.append(f"{'ckpt_best':>14s}")

    header = " | ".join(header_parts)
    logger.log(header)
    logger.log("-" * len(header))

    for name, fields, metrics in results:
        row_parts = [f"{name:40s}"]
        row_parts.append(f"{metrics['mean_l2']:12.4e}")

        for f in all_fields:
            l2_val = metrics.get(f"L2_{f}")
            rmse_val = metrics.get(f"RMSE_{f}")
            row_parts.append(f"{l2_val:14.4e}" if isinstance(l2_val, float) else f"{'N/A':>14s}")
            row_parts.append(f"{rmse_val:14.4e}" if isinstance(rmse_val, float) else f"{'N/A':>14s}")

        a_mean = metrics.get("active_mean_l2", float('nan'))
        i_mean = metrics.get("inactive_mean_l2", float('nan'))
        row_parts.append(f"{a_mean:14.4e}" if not math.isnan(a_mean) else f"{'N/A':>14s}")
        row_parts.append(f"{i_mean:14.4e}" if not math.isnan(i_mean) else f"{'N/A':>14s}")

        best_ve = metrics.get("_best_val_error", "N/A")
        if isinstance(best_ve, float):
            row_parts.append(f"{best_ve:14.4e}")
        else:
            row_parts.append(f"{str(best_ve):>14s}")

        logger.log(" | ".join(row_parts))


# ============================================================
# Main
# ============================================================

def main():
    if not CONFIG_LIST:
        print("ERROR: CONFIG_LIST is empty. Please edit evaluate.py and add config paths.")
        return

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"result_keyhole/evaluate/report_{timestamp}.txt"
    logger = DualLogger(report_path)

    logger.log(f"{'#'*70}")
    logger.log(f"# Evaluate Report")
    logger.log(f"# Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"# Device : {device_str}")
    logger.log(f"# Configs: {len(CONFIG_LIST)}")
    logger.log(f"# Report : {report_path}")
    logger.log(f"{'#'*70}")

    results = []      # list of (name, fields, metrics)
    failures = []     # list of (config_path, error_msg)

    for idx, config_path in enumerate(CONFIG_LIST):
        logger.log(f"\n>>> [{idx+1}/{len(CONFIG_LIST)}] {config_path}")

        if not os.path.exists(config_path):
            msg = f"Config file not found: {config_path}"
            logger.log(f"  [FAIL] {msg}")
            failures.append((config_path, msg))
            continue

        try:
            result = evaluate_single(config_path, device, logger)
            if result is not None:
                results.append(result)
            else:
                failures.append((config_path, "Skipped (checkpoint not found)"))
        except Exception as e:
            tb = traceback.format_exc()
            logger.log(f"  [FAIL] Exception: {e}")
            logger.log(f"  Traceback:\n{tb}")
            failures.append((config_path, str(e)))

        # 释放显存
        torch.cuda.empty_cache()

    # 汇总
    print_summary_table(logger, results)

    # 失败列表
    if failures:
        logger.log(f"\n{'='*70}")
        logger.log(f"FAILURES ({len(failures)}):")
        logger.log(f"{'='*70}")
        for cfg, reason in failures:
            logger.log(f"  {cfg}")
            logger.log(f"    Reason: {reason}")

    logger.log(f"\n{'#'*70}")
    logger.log(f"# Done. {len(results)} succeeded, {len(failures)} failed.")
    logger.log(f"# Report saved to: {report_path}")
    logger.log(f"{'#'*70}")

    logger.close()


if __name__ == "__main__":
    main()
