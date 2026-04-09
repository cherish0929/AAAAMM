#!/usr/bin/env python3
"""
analyze_score_distribution.py
=============================
Analyze the distribution of the physics-based scoring used in the adaptive
graph pipeline:  T_score, interface_score, and final combined score.

For each component, outputs:
  - Basic statistics (mean, std, min, max, median, percentiles)
  - Histogram data (bin counts across [0, 1])
  - ASCII histogram for quick visual inspection

Optionally saves matplotlib histograms to result/ if matplotlib is available.

Usage:
    python analyze_score_distribution.py --config config/adaptive_graph_compressed.json
    python analyze_score_distribution.py --config config/adaptive_graph_test.json --n_samples 4 --save_plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from src.adaptive_graph import (
    _FIELD_STATS,
    _physics_score,
    _smooth_ramp,
    _safe_normalize,
    AdaptiveGraphManager,
    build_backbone_edges,
    build_backbone_indices,
    classify_zones,
    compute_active_score,
)
from src.dataset_adaptive import AdaptiveGraphDataset
from src.utils import load_json_config, set_seed


# ======================================================================
#  Score decomposition: compute each component separately
# ======================================================================

@torch.no_grad()
def decompose_score(
    pred_field: torch.Tensor,
    gt_field: Optional[torch.Tensor],
    prev_field: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    fields: List[str],
    T_ref: float = 600.0,
    T_high: float = 1500.0,
    interface_bonus: float = 0.5,
    air_discount: float = 0.5,
    gt_blend: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute each score component individually and return all of them.

    Returns dict with keys:
        'T_score'          - temperature-based score [0,1]
        'interface_score'  - alpha.air interface score [0,1]
        'air_multiplier'   - 1.0 for non-gas, air_discount for pure gas
        'final_score'      - combined score [0,1]
    """
    if gt_field is not None and gt_blend > 0.0:
        field = gt_blend * gt_field + (1.0 - gt_blend) * pred_field
    else:
        field = pred_field

    stats = _FIELD_STATS
    has_T = "T" in fields
    has_alpha = "alpha.air" in fields

    result = OrderedDict()

    if has_T and has_alpha:
        T_idx = fields.index("T")
        alpha_idx = fields.index("alpha.air")

        T_norm = field[:, T_idx]
        alpha_norm = field[:, alpha_idx]

        T_mean, T_std = stats.get("T", (5.2999e+02, 4.5454e+02))
        T_ref_norm = (T_ref - T_mean) / T_std if T_std != 0 else T_ref - T_mean
        T_high_norm = (T_high - T_mean) / T_std if T_std != 0 else T_high - T_mean

        T_score = _smooth_ramp(T_norm, T_ref_norm, T_high_norm)

        alpha_mean, alpha_std = stats.get("alpha.air", (0, 1))
        alpha_raw = alpha_norm * alpha_std + alpha_mean
        alpha_raw = alpha_raw.clamp(0.0, 1.0)

        interface_score = 4.0 * alpha_raw * (1.0 - alpha_raw)

        is_air = (alpha_raw >= 0.9).float()
        air_mult = 1.0 - (1.0 - air_discount) * is_air

        final = (T_score * (1.0 + interface_bonus * interface_score) * air_mult).clamp(0.0, 1.0)

        result["T_score"] = T_score
        result["interface_score"] = interface_score
        result["air_multiplier"] = air_mult
        result["final_score"] = final
    else:
        temporal = (field - prev_field).abs().mean(dim=-1)
        result["temporal_change"] = _safe_normalize(temporal)
        result["final_score"] = result["temporal_change"]

    return result


# ======================================================================
#  Statistics helpers
# ======================================================================

def compute_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Compute descriptive statistics for a 1-D tensor."""
    t = tensor.float().cpu()
    n = t.numel()
    if n == 0:
        return {k: 0.0 for k in [
            "count", "mean", "std", "min", "max",
            "p5", "p25", "median", "p75", "p95",
            "pct_zero", "pct_below_01", "pct_above_09",
            "skewness",
        ]}

    sorted_t, _ = t.sort()
    mean = float(t.mean())
    std = float(t.std())

    def _percentile(p):
        idx = int(p / 100.0 * (n - 1))
        return float(sorted_t[idx])

    pct_zero = float((t == 0).sum()) / n * 100
    pct_below_01 = float((t < 0.1).sum()) / n * 100
    pct_above_09 = float((t > 0.9).sum()) / n * 100

    # skewness
    if std > 1e-12:
        skew = float(((t - mean) ** 3).mean() / (std ** 3))
    else:
        skew = 0.0

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": float(t.min()),
        "max": float(t.max()),
        "p5": _percentile(5),
        "p25": _percentile(25),
        "median": _percentile(50),
        "p75": _percentile(75),
        "p95": _percentile(95),
        "pct_zero": pct_zero,
        "pct_below_01": pct_below_01,
        "pct_above_09": pct_above_09,
        "skewness": skew,
    }


def histogram_counts(tensor: torch.Tensor, n_bins: int = 20) -> Tuple[List[int], List[float]]:
    """Compute histogram bin counts over [0, 1]."""
    t = tensor.float().cpu().clamp(0.0, 1.0)
    edges = np.linspace(0, 1, n_bins + 1)
    counts = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            c = int(((t >= lo) & (t <= hi)).sum())
        else:
            c = int(((t >= lo) & (t < hi)).sum())
        counts.append(c)
    bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    return counts, bin_centers


def ascii_histogram(counts: List[int], bin_centers: List[float], width: int = 50) -> str:
    """Render a simple ASCII histogram."""
    max_count = max(counts) if counts else 1
    lines = []
    for center, count in zip(bin_centers, counts):
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_len
        lines.append(f"  {center:>5.2f} | {bar:<{width}} {count:>8,}")
    return "\n".join(lines)


# ======================================================================
#  Main analysis
# ======================================================================

def analyze_sample(
    batch: dict,
    adaptive_cfg: dict,
    device: torch.device,
    epoch: int = 0,
    horizon: int = 5,
    fields: list = None,
) -> List[Dict[str, Dict]]:
    """Analyze score distributions for each refresh step of one sample.

    Returns a list (one per refresh step) of dicts mapping
    component_name -> { stats: {...}, histogram_counts: [...], bin_centers: [...] }
    """
    state = batch["state"].to(device)
    fullres_pos = batch["fullres_pos"].to(device)
    backbone_indices = batch["backbone_indices"]
    backbone_shape = batch["backbone_shape"]
    backbone_edges = batch["backbone_edges"].to(device)
    fullres_shape = batch["fullres_shape"]

    fp = fullres_pos[0]
    bb_idx_np = backbone_indices[0].cpu().numpy()
    bb_shape = tuple(backbone_shape[0].tolist())
    bb_edges = backbone_edges[0]
    fr_shape = tuple(fullres_shape[0].tolist())

    mgr = AdaptiveGraphManager(
        cfg=adaptive_cfg,
        backbone_indices=bb_idx_np,
        backbone_shape=bb_shape,
        backbone_edges=bb_edges,
        fullres_pos=fp,
        fullres_grid_shape=fr_shape,
        device=device,
        fields=fields or [],
    )

    az_cfg = mgr.az_cfg
    refresh_K = mgr.refresh_K

    full_state = state[0, 0].clone()
    prev_state = full_state.clone()
    T = min(horizon, state.shape[1] - 1)

    results = []

    for t in range(T):
        gt_field = state[0, t + 1] if (t + 1) < state.shape[1] else None
        is_refresh = (t % refresh_K == 0)

        if not is_refresh:
            if gt_field is not None:
                prev_state = full_state.clone()
                full_state = gt_field.clone()
            continue

        gt_blend = mgr.get_gt_blend(epoch)

        components = decompose_score(
            pred_field=full_state,
            gt_field=gt_field,
            prev_field=prev_state,
            grid_shape=fr_shape,
            fields=fields or [],
            T_ref=az_cfg.get("T_ref", 600.0),
            T_high=az_cfg.get("T_high", 1500.0),
            interface_bonus=az_cfg.get("interface_bonus", 0.5),
            air_discount=az_cfg.get("air_discount", 0.5),
            gt_blend=gt_blend,
        )

        step_result = {"timestep": t, "gt_blend": gt_blend}
        for name, tensor in components.items():
            stats = compute_stats(tensor)
            counts, centers = histogram_counts(tensor, n_bins=20)
            step_result[name] = {
                "stats": stats,
                "histogram_counts": counts,
                "bin_centers": centers,
            }

        # Also compute zone distribution based on final score
        zone = classify_zones(
            components["final_score"],
            core_threshold=az_cfg.get("core_threshold", 0.6),
            ring_threshold=az_cfg.get("ring_threshold", 0.2),
        )
        n_full = full_state.shape[0]
        step_result["zone_counts"] = {
            "core": int((zone == 2).sum().item()),
            "ring": int((zone == 1).sum().item()),
            "background": int((zone == 0).sum().item()),
            "total": n_full,
        }

        results.append(step_result)

        if gt_field is not None:
            prev_state = full_state.clone()
            full_state = gt_field.clone()

    return results


def format_component_report(name: str, data: dict) -> str:
    """Format the analysis of one score component."""
    stats = data["stats"]
    counts = data["histogram_counts"]
    centers = data["bin_centers"]

    lines = []
    lines.append(f"  --- {name} ---")
    lines.append(f"    N={stats['count']:,}  mean={stats['mean']:.6f}  std={stats['std']:.6f}")
    lines.append(f"    min={stats['min']:.6f}  max={stats['max']:.6f}")
    lines.append(f"    percentiles: p5={stats['p5']:.4f}  p25={stats['p25']:.4f}  "
                 f"median={stats['median']:.4f}  p75={stats['p75']:.4f}  p95={stats['p95']:.4f}")
    lines.append(f"    pct_zero={stats['pct_zero']:.1f}%  pct<0.1={stats['pct_below_01']:.1f}%  "
                 f"pct>0.9={stats['pct_above_09']:.1f}%")
    lines.append(f"    skewness={stats['skewness']:.4f}"
                 f"  ({'right-skewed / mass near 0' if stats['skewness'] > 0.5 else 'left-skewed / mass near 1' if stats['skewness'] < -0.5 else 'roughly symmetric'})")
    lines.append("")
    lines.append(f"    Histogram [0, 1]:")
    lines.append(ascii_histogram(counts, centers, width=40))
    lines.append("")
    return "\n".join(lines)


def try_save_plots(all_results: list, output_dir: str):
    """Try to save matplotlib histograms. Silently skip if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot generation)")
        return

    component_names = ["T_score", "interface_score", "air_multiplier", "final_score"]

    # Individual plots
    for comp_name in component_names:
        all_counts = []
        all_centers = None
        for sample_results in all_results:
            for step in sample_results:
                if comp_name not in step:
                    continue
                data = step[comp_name]
                all_counts.append(data["histogram_counts"])
                if all_centers is None:
                    all_centers = data["bin_centers"]

        if not all_counts or all_centers is None:
            continue

        avg_counts = np.mean(all_counts, axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = all_centers[1] - all_centers[0] if len(all_centers) > 1 else 0.05
        ax.bar(all_centers, avg_counts, width=bar_width * 0.9, alpha=0.75, color="#4a90d9")
        ax.set_xlabel("Score Value", fontsize=12)
        ax.set_ylabel("Average Count", fontsize=12)
        ax.set_title(f"Distribution: {comp_name}", fontsize=14)
        ax.set_xlim(-0.02, 1.02)

        all_stats_vals = []
        for sample_results in all_results:
            for step in sample_results:
                if comp_name in step:
                    all_stats_vals.append(step[comp_name]["stats"])

        if all_stats_vals:
            avg_mean = np.mean([s["mean"] for s in all_stats_vals])
            avg_std = np.mean([s["std"] for s in all_stats_vals])
            avg_median = np.mean([s["median"] for s in all_stats_vals])
            avg_pct_zero = np.mean([s["pct_zero"] for s in all_stats_vals])
            avg_pct_below_01 = np.mean([s["pct_below_01"] for s in all_stats_vals])
            avg_pct_above_09 = np.mean([s["pct_above_09"] for s in all_stats_vals])

            text = (f"mean={avg_mean:.4f}, std={avg_std:.4f}\n"
                    f"median={avg_median:.4f}\n"
                    f"pct=0: {avg_pct_zero:.1f}%\n"
                    f"pct<0.1: {avg_pct_below_01:.1f}%\n"
                    f"pct>0.9: {avg_pct_above_09:.1f}%")
            ax.text(0.98, 0.95, text, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"score_dist_{comp_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # Summary subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ["#4a90d9", "#e67e22", "#2ecc71", "#e74c3c"]

    for i, comp_name in enumerate(component_names):
        ax = axes[i]
        all_counts = []
        all_centers_c = None
        for sample_results in all_results:
            for step in sample_results:
                if comp_name in step:
                    all_counts.append(step[comp_name]["histogram_counts"])
                    if all_centers_c is None:
                        all_centers_c = step[comp_name]["bin_centers"]

        if not all_counts or all_centers_c is None:
            ax.set_title(f"{comp_name} (no data)")
            continue

        avg_counts = np.mean(all_counts, axis=0)
        bar_w = all_centers_c[1] - all_centers_c[0] if len(all_centers_c) > 1 else 0.05
        ax.bar(all_centers_c, avg_counts, width=bar_w * 0.9, alpha=0.75, color=colors[i % len(colors)])
        ax.set_title(comp_name, fontsize=13)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Score")
        ax.set_ylabel("Avg Count")

    plt.suptitle("Physics-Based Score Components", fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "score_dist_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze active score component distributions")
    parser.add_argument("--config", type=str, default="config/adaptive_graph_compressed.json")
    parser.add_argument("--n_samples", type=int, default=2,
                        help="Number of data samples to analyze")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override horizon (default: from config)")
    parser.add_argument("--epoch", type=int, default=0,
                        help="Simulated epoch for gt_blend decay")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_plots", action="store_true",
                        help="Save matplotlib histogram plots to result/")
    cli = parser.parse_args()

    args = load_json_config(cli.config)
    if cli.seed is not None:
        set_seed(cli.seed)

    device_str = cli.device or getattr(args, "device", "cpu")
    if "cuda" in device_str and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    data_cfg = args.data
    adaptive_cfg = getattr(args, "adaptive_graph", None) or {}
    horizon = cli.horizon or data_cfg.get("horizon_train", 5)

    print("=" * 78)
    print(f"  Score Distribution Analysis  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: {cli.config}")
    print(f"  Device: {device_str}  |  Epoch: {cli.epoch}  |  Horizon: {horizon}")
    print("=" * 78)
    print()

    # Print scoring params
    az_cfg = adaptive_cfg.get("active_zone", {})
    print("[Scoring Parameters]")
    print(f"  T_ref            = {az_cfg.get('T_ref', 600.0)}")
    print(f"  T_high           = {az_cfg.get('T_high', 1500.0)}")
    print(f"  interface_bonus  = {az_cfg.get('interface_bonus', 0.5)}")
    print(f"  air_discount     = {az_cfg.get('air_discount', 0.5)}")
    print(f"  core_threshold   = {az_cfg.get('core_threshold', 0.6)}")
    print(f"  ring_threshold   = {az_cfg.get('ring_threshold', 0.2)}")
    print()

    # Load dataset
    print("[Loading Dataset...]")
    common_kwargs = dict(
        data_cfg=data_cfg,
        fields=data_cfg.get("fields", ["T"]),
        input_steps=data_cfg.get("input_steps", 1),
        time_stride=data_cfg.get("time_stride", 1),
        spatial_stride=data_cfg.get("spatial_stride", 1),
        normalize=data_cfg.get("normalize", True),
        samples_per_file=data_cfg.get("samples_per_file", 32),
        norm_cache=data_cfg.get("norm_cache"),
        adaptive_cfg=adaptive_cfg,
    )
    dataset = AdaptiveGraphDataset(
        file_list=data_cfg["train_list"],
        mode="train",
        horizon=horizon,
        **common_kwargs,
    )
    print(f"  Dataset: {len(dataset)} samples")
    print()

    # Analyze
    all_results = []
    n_samples = min(cli.n_samples, len(dataset))

    for sample_idx in range(n_samples):
        print(f"{'=' * 78}")
        print(f"  Sample {sample_idx + 1} / {n_samples}")
        print(f"{'=' * 78}")
        print()

        batch_raw = dataset[sample_idx]
        batch = {}
        for k, v in batch_raw.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
            else:
                batch[k] = v

        sample_results = analyze_sample(
            batch=batch,
            adaptive_cfg=adaptive_cfg,
            device=device,
            epoch=cli.epoch,
            horizon=horizon,
            fields=data_cfg.get("fields", []),
        )

        for step_data in sample_results:
            t = step_data["timestep"]
            print(f"  Refresh t={t}  (gt_blend={step_data['gt_blend']:.3f})")
            print()

            for comp_name in [
                "T_score", "solid_score", "interface_score", "final_score",
            ]:
                if comp_name in step_data:
                    print(format_component_report(comp_name, step_data[comp_name]))

            zc = step_data["zone_counts"]
            total = zc["total"]
            print(f"  Zone Distribution (from final_score):")
            print(f"    Core:       {zc['core']:>8,}  ({zc['core']/total*100:>5.1f}%)")
            print(f"    Ring:       {zc['ring']:>8,}  ({zc['ring']/total*100:>5.1f}%)")
            print(f"    Background: {zc['background']:>8,}  ({zc['background']/total*100:>5.1f}%)")
            print()

        all_results.append(sample_results)

    # ---- Aggregate across all samples ----
    print("=" * 78)
    print("  AGGREGATE (averaged over all samples & refresh steps)")
    print("=" * 78)
    print()

    component_names = [
        "T_score", "solid_score", "interface_score", "final_score",
    ]

    for comp_name in component_names:
        all_stats = []
        for sample_results in all_results:
            for step in sample_results:
                if comp_name in step:
                    all_stats.append(step[comp_name]["stats"])

        if not all_stats:
            continue

        print(f"  --- {comp_name} (aggregate) ---")
        for key in ["mean", "std", "median", "pct_zero", "pct_below_01", "pct_above_09", "skewness"]:
            vals = [s[key] for s in all_stats]
            avg = np.mean(vals)
            if key.startswith("pct"):
                print(f"    {key:<20} = {avg:>8.1f}%")
            else:
                print(f"    {key:<20} = {avg:>8.6f}")
        print()

    # ---- Save plots ----
    if cli.save_plots:
        result_dir = os.path.join(str(PROJ_ROOT), "result")
        os.makedirs(result_dir, exist_ok=True)
        print("[Saving plots...]")
        try_save_plots(all_results, result_dir)
        print()

    # ---- Save text report ----
    result_dir = os.path.join(str(PROJ_ROOT), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(result_dir, f"score_distribution_{timestamp}.txt")

    report_lines = []
    report_lines.append(f"Score Distribution Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Config: {cli.config}")
    report_lines.append(f"Samples: {n_samples}, Horizon: {horizon}, Epoch: {cli.epoch}")
    report_lines.append("")
    report_lines.append(json.dumps(az_cfg, indent=2))
    report_lines.append("")

    for sample_idx, sample_results in enumerate(all_results):
        for step_data in sample_results:
            report_lines.append(f"Sample {sample_idx}, t={step_data['timestep']}")
            for comp_name in component_names:
                if comp_name in step_data:
                    report_lines.append(format_component_report(comp_name, step_data[comp_name]))

    with open(out_file, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to: {out_file}")


if __name__ == "__main__":
    main()
