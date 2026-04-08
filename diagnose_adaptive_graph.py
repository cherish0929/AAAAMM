#!/usr/bin/env python3
"""
diagnose_adaptive_graph.py
==========================
Standalone diagnostic script for the zone-first adaptive graph pipeline.

Reads real data samples, runs the adaptive graph construction flow
(scoring → zoning → sampling → edge building) WITHOUT training,
and prints comprehensive statistics about:
  - Graph structure per refresh step (nodes, edges, zones)
  - Active-first effectiveness (zone connectivity, density)
  - Performance profiling (timing of each stage)

Usage:
    python diagnose_adaptive_graph.py --config config/adaptive_graph_new.json
    python diagnose_adaptive_graph.py --config config/adaptive_graph_new.json --n_samples 2 --horizon 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# -- add project root to path --
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from src.adaptive_graph import (
    AdaptiveGraphManager,
    build_backbone_edges,
    build_backbone_indices,
    build_stencil_edges,
    build_stencil_edges_compressed,
    canonicalize_and_dedup_edges,
    classify_zones,
    compute_active_score,
    enforce_edge_budget,
    enforce_per_node_degree,
    zone_first_sample,
)
from src.dataset_adaptive import AdaptiveGraphDataset
from src.utils import ChannelNormalizer, load_json_config, set_seed


# ======================================================================
#  Helper: Timer context manager
# ======================================================================
class Timer:
    """Simple context-manager timer that accumulates times."""
    def __init__(self):
        self.records: Dict[str, List[float]] = defaultdict(list)
        self._stack = []

    def __call__(self, name: str):
        self._current_name = name
        return self

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._stack.append((self._current_name, time.perf_counter()))
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        name, t0 = self._stack.pop()
        self.records[name].append(time.perf_counter() - t0)

    def summary(self) -> str:
        lines = []
        lines.append(f"{'Stage':<35} {'Calls':>6} {'Total(s)':>10} {'Mean(ms)':>10} {'Max(ms)':>10}")
        lines.append("-" * 75)
        for name, times in self.records.items():
            total = sum(times)
            mean_ms = total / len(times) * 1000
            max_ms = max(times) * 1000
            lines.append(f"{name:<35} {len(times):>6} {total:>10.4f} {mean_ms:>10.2f} {max_ms:>10.2f}")
        return "\n".join(lines)


# ======================================================================
#  Diagnostic core
# ======================================================================

def diagnose_single_sample(
    batch: dict,
    adaptive_cfg: dict,
    device: torch.device,
    timer: Timer,
    epoch: int = 0,
    horizon: int = 5,
    fields: list = None,
) -> List[dict]:
    """Run adaptive graph refresh on one sample and collect stats per refresh step.

    Returns a list of dicts, one per refresh event.
    """
    state = batch["state"].to(device)                       # [1, 1+H, N_full, C]
    fullres_pos = batch["fullres_pos"].to(device)           # [1, N_full, 3]
    backbone_indices = batch["backbone_indices"]             # [1, N_bb]
    backbone_shape = batch["backbone_shape"]                 # [1, 3]
    backbone_edges = batch["backbone_edges"].to(device)     # [1, E_bb, 2]
    fullres_shape = batch["fullres_shape"]                   # [1, 3]

    fp = fullres_pos[0]
    bb_idx_np = backbone_indices[0].cpu().numpy()
    bb_shape = tuple(backbone_shape[0].tolist())
    bb_edges = backbone_edges[0]
    fr_shape = tuple(fullres_shape[0].tolist())

    N_full = fp.shape[0]
    N_bb = len(bb_idx_np)

    # init manager
    with timer("mgr_init"):
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

    full_state = state[0, 0].clone()   # [N_full, C]
    prev_state = full_state.clone()
    T = min(horizon, state.shape[1] - 1)
    refresh_K = mgr.refresh_K

    results = []

    # Check for compressed mode
    compress_cfg = adaptive_cfg.get("compression", {})
    use_compressed = compress_cfg.get("enabled", False)

    for t in range(T):
        gt_field = state[0, t + 1] if (t + 1) < state.shape[1] else None
        is_refresh = (t % refresh_K == 0)

        if not is_refresh:
            # just advance state (simulate stepping without actual model forward)
            if gt_field is not None:
                prev_state = full_state.clone()
                full_state = gt_field.clone()
            continue

        # --- Detailed refresh with timing ---
        gt_blend = mgr.get_gt_blend(epoch)
        az_cfg = mgr.az_cfg

        # Step 1: Active score
        with timer("active_score"):
            score = compute_active_score(
                pred_field=full_state,
                gt_field=gt_field,
                prev_field=prev_state,
                grid_shape=fr_shape,
                gradient_weight=az_cfg.get("gradient_weight", 0.5),
                temporal_weight=az_cfg.get("temporal_weight", 0.3),
                physics_weight=az_cfg.get("physics_weight", 0.2),
                gt_blend=gt_blend,
                physics_triggers=mgr._physics_triggers,
            )

        # Step 2: Zone classification
        with timer("zoning"):
            zone = classify_zones(
                score,
                core_threshold=az_cfg.get("core_threshold", 0.6),
                ring_threshold=az_cfg.get("ring_threshold", 0.2),
            )

        # Step 3: Sampling
        with timer("sampling"):
            selected = zone_first_sample(
                zone_full=zone,
                backbone_mask=mgr.backbone_mask,
                core_keep_ratio=az_cfg.get("core_keep_ratio", 1.0),
                ring_keep_ratio=az_cfg.get("ring_keep_ratio", 0.5),
                background_extra_ratio=az_cfg.get("background_extra_ratio", 0.0),
                bg_backbone_keep_ratio=az_cfg.get("bg_backbone_keep_ratio", 1.0),
            )

        # Step 4: Stencil edges
        edge_cfg = mgr.edge_cfg
        with timer("edge_construction"):
            if use_compressed:
                intra_edges = build_stencil_edges_compressed(
                    selected_indices=selected,
                    zone_labels=zone,
                    grid_shape=fr_shape,
                    backbone_mask=mgr.backbone_mask,
                    core_max_neighbors=compress_cfg.get("core_max_neighbors", 12),
                    ring_max_neighbors=compress_cfg.get("ring_max_neighbors", 8),
                    bg_keep_ratio=compress_cfg.get("bg_edge_keep_ratio", 0.7),
                )
            else:
                intra_edges = build_stencil_edges(
                    selected_indices=selected,
                    zone_labels=zone,
                    grid_shape=fr_shape,
                    backbone_mask=mgr.backbone_mask,
                    bg_stencil=edge_cfg.get("bg_stencil", 6),
                    ring_stencil=edge_cfg.get("ring_stencil", 18),
                    core_stencil=edge_cfg.get("core_stencil", 26),
                )

        # Step 5: Cross-layer edges
        N_full_int = mgr.n_full
        full_to_local = torch.full((N_full_int,), -1, dtype=torch.long, device=device)
        full_to_local[selected] = torch.arange(selected.shape[0], device=device)

        with timer("cross_layer_edges"):
            from src.adaptive_graph import build_cross_layer_edges
            cross_edges = build_cross_layer_edges(
                selected_indices=selected,
                zone_labels=zone,
                backbone_indices_set=mgr.bb_idx_tensor,
                backbone_stride=mgr.backbone_stride,
                grid_shape=fr_shape,
                full_to_local=full_to_local,
            )

        # Combine edges (with compression pipeline if enabled)
        edge_parts = []
        if intra_edges.numel() > 0:
            edge_parts.append(intra_edges)
        if cross_edges.numel() > 0:
            if use_compressed:
                cross_edges = canonicalize_and_dedup_edges(cross_edges)
            edge_parts.append(cross_edges)

        if edge_parts:
            all_edges = torch.cat(edge_parts, dim=0)
            if use_compressed:
                all_edges = canonicalize_and_dedup_edges(all_edges)
        else:
            all_edges = torch.zeros((0, 2), dtype=torch.long, device=device)

        # Enforce global edge budget (compressed mode)
        if use_compressed and all_edges.numel() > 0:
            sel_zone_for_budget = zone[selected]
            max_edges_ratio = compress_cfg.get("max_edges_ratio", 4.0)
            all_edges = enforce_edge_budget(
                edges=all_edges,
                zone_labels_selected=sel_zone_for_budget,
                max_edges_ratio=max_edges_ratio,
                num_nodes=selected.shape[0],
            )

        # Enforce per-node degree limits (compressed mode)
        if use_compressed and all_edges.numel() > 0:
            sel_zone_for_deg = zone[selected]
            all_edges = enforce_per_node_degree(
                edges=all_edges,
                zone_labels_selected=sel_zone_for_deg,
                core_max_degree=compress_cfg.get("core_max_neighbors", 12),
                ring_max_degree=compress_cfg.get("ring_max_neighbors", 8),
                bg_max_degree=int(6 * compress_cfg.get("bg_edge_keep_ratio", 0.7)),
                num_nodes=selected.shape[0],
            )

        # ---- Collect statistics ----
        N_sel = selected.shape[0]
        E_total = all_edges.shape[0]
        E_intra = intra_edges.shape[0]
        E_cross = cross_edges.shape[0] if cross_edges.numel() > 0 else 0

        # Zone counts (full grid)
        n_core_full = int((zone == 2).sum().item())
        n_ring_full = int((zone == 1).sum().item())
        n_bg_full = int((zone == 0).sum().item())

        # Zone counts (selected subgraph)
        sel_zone = zone[selected]
        n_core_sel = int((sel_zone == 2).sum().item())
        n_ring_sel = int((sel_zone == 1).sum().item())
        n_bg_sel = int((sel_zone == 0).sum().item())

        # Backbone nodes in active zones
        bb_mask_sel = mgr.backbone_mask[selected]
        bb_in_core = int(((sel_zone == 2) & bb_mask_sel).sum().item())
        bb_in_ring = int(((sel_zone == 1) & bb_mask_sel).sum().item())
        bb_in_bg = int(((sel_zone == 0) & bb_mask_sel).sum().item())
        bb_in_active = bb_in_core + bb_in_ring

        # Edge analysis by zone
        if E_total > 0:
            src_zone = sel_zone[all_edges[:, 0]]
            dst_zone = sel_zone[all_edges[:, 1]]

            # core-related edges: at least one endpoint is core
            core_edge_mask = (src_zone == 2) | (dst_zone == 2)
            n_core_edges = int(core_edge_mask.sum().item())

            # ring-related edges: at least one endpoint is ring (exclude pure core-core)
            ring_edge_mask = ((src_zone == 1) | (dst_zone == 1))
            n_ring_edges = int(ring_edge_mask.sum().item())

            # background-only edges: both endpoints are background
            bg_edge_mask = (src_zone == 0) & (dst_zone == 0)
            n_bg_edges = int(bg_edge_mask.sum().item())

            # backbone-background edges (both in bg zone, at least one is backbone)
            src_is_bb = mgr.backbone_mask[selected[all_edges[:, 0]]]
            dst_is_bb = mgr.backbone_mask[selected[all_edges[:, 1]]]
            bb_bg_mask = bg_edge_mask & (src_is_bb | dst_is_bb)
            n_bb_bg_edges = int(bb_bg_mask.sum().item())

            # Per-zone average degree
            degree = torch.zeros(N_sel, device=device)
            flat_edges = all_edges.view(-1)
            degree.scatter_add_(0, flat_edges, torch.ones(flat_edges.shape[0], device=device))
            # degree counted twice (src + dst), so this gives total degree

            core_deg = degree[sel_zone == 2]
            ring_deg = degree[sel_zone == 1]
            bg_deg = degree[sel_zone == 0]

            avg_core_deg = float(core_deg.mean().item()) if core_deg.numel() > 0 else 0.0
            avg_ring_deg = float(ring_deg.mean().item()) if ring_deg.numel() > 0 else 0.0
            avg_bg_deg = float(bg_deg.mean().item()) if bg_deg.numel() > 0 else 0.0
        else:
            n_core_edges = n_ring_edges = n_bg_edges = n_bb_bg_edges = 0
            avg_core_deg = avg_ring_deg = avg_bg_deg = 0.0

        # Score statistics
        score_core = score[zone == 2]
        score_ring = score[zone == 1]
        score_bg = score[zone == 0]

        # Edge ratio (edges per node)
        edges_per_node = E_total / max(N_sel, 1)

        # Check canonicality (for compressed mode)
        n_canonical = 0
        n_unique_after_canon = E_total
        if E_total > 0:
            src_e = all_edges[:, 0]
            dst_e = all_edges[:, 1]
            n_canonical = int((src_e < dst_e).sum().item())
            canon_src = torch.min(src_e, dst_e)
            canon_dst = torch.max(src_e, dst_e)
            canon_packed = canon_src * (N_sel + 1) + canon_dst
            n_unique_after_canon = int(torch.unique(canon_packed).shape[0])

        stats = {
            "timestep": t,
            "refresh_K": refresh_K,
            "gt_blend": gt_blend,
            # Full grid
            "N_full": N_full,
            "N_backbone": N_bb,
            "full_grid_shape": fr_shape,
            "backbone_shape": bb_shape,
            # Zone counts (full grid)
            "n_core_full": n_core_full,
            "n_ring_full": n_ring_full,
            "n_bg_full": n_bg_full,
            "core_pct_full": n_core_full / N_full * 100,
            "ring_pct_full": n_ring_full / N_full * 100,
            "bg_pct_full": n_bg_full / N_full * 100,
            # Selected subgraph
            "N_selected": N_sel,
            "n_core_sel": n_core_sel,
            "n_ring_sel": n_ring_sel,
            "n_bg_sel": n_bg_sel,
            "core_pct_sel": n_core_sel / max(N_sel, 1) * 100,
            "ring_pct_sel": n_ring_sel / max(N_sel, 1) * 100,
            "bg_pct_sel": n_bg_sel / max(N_sel, 1) * 100,
            # Compression
            "node_compression": N_sel / N_full,
            "full_grid_edges_6nn": N_full * 3,  # approximate max 6-neighbor edges
            "E_total": E_total,
            "E_intra": E_intra,
            "E_cross": E_cross,
            # Edge compression stats
            "edges_per_node": edges_per_node,
            "n_canonical": n_canonical,
            "n_unique_after_canon": n_unique_after_canon,
            "is_compressed": use_compressed,
            # Backbone in active zones
            "bb_in_core": bb_in_core,
            "bb_in_ring": bb_in_ring,
            "bb_in_bg": bb_in_bg,
            "bb_in_active": bb_in_active,
            "bb_in_active_pct": bb_in_active / max(N_bb, 1) * 100,
            # Edge analysis
            "n_core_edges": n_core_edges,
            "n_ring_edges": n_ring_edges,
            "n_bg_edges": n_bg_edges,
            "n_bb_bg_edges": n_bb_bg_edges,
            # Connectivity / density
            "avg_core_degree": avg_core_deg,
            "avg_ring_degree": avg_ring_deg,
            "avg_bg_degree": avg_bg_deg,
            "core_edge_density": n_core_edges / max(n_core_sel, 1),
            "ring_edge_density": n_ring_edges / max(n_ring_sel, 1),
            "bg_edge_density": n_bg_edges / max(n_bg_sel, 1),
            # Score stats
            "score_mean": float(score.mean().item()),
            "score_std": float(score.std().item()),
            "score_core_mean": float(score_core.mean().item()) if score_core.numel() > 0 else 0.0,
            "score_ring_mean": float(score_ring.mean().item()) if score_ring.numel() > 0 else 0.0,
            "score_bg_mean": float(score_bg.mean().item()) if score_bg.numel() > 0 else 0.0,
        }
        results.append(stats)

        # advance state for next step
        if gt_field is not None:
            prev_state = full_state.clone()
            full_state = gt_field.clone()

    return results


def format_stats(stats: dict) -> str:
    """Format a single refresh step's stats into human-readable text."""
    lines = []
    lines.append(f"  ---- Refresh at t={stats['timestep']} (K={stats['refresh_K']}, gt_blend={stats['gt_blend']:.3f}) ----")
    lines.append("")

    # Grid info
    lines.append(f"  Full Grid: {stats['full_grid_shape']}  => N_full = {stats['N_full']:,}")
    lines.append(f"  Backbone:  {stats['backbone_shape']}  => N_bb   = {stats['N_backbone']:,}")
    lines.append("")

    # Zone distribution (full grid)
    lines.append(f"  Zone Distribution (full grid):")
    lines.append(f"    Core (zone=2):  {stats['n_core_full']:>8,}  ({stats['core_pct_full']:>5.1f}%)")
    lines.append(f"    Ring (zone=1):  {stats['n_ring_full']:>8,}  ({stats['ring_pct_full']:>5.1f}%)")
    lines.append(f"    Background:     {stats['n_bg_full']:>8,}  ({stats['bg_pct_full']:>5.1f}%)")
    lines.append("")

    # Selected subgraph
    lines.append(f"  Selected Subgraph: N_sel = {stats['N_selected']:,}  (compression: {stats['node_compression']:.4f} = {stats['node_compression']*100:.1f}%)")
    lines.append(f"    Core:       {stats['n_core_sel']:>8,}  ({stats['core_pct_sel']:>5.1f}% of selected)")
    lines.append(f"    Ring:       {stats['n_ring_sel']:>8,}  ({stats['ring_pct_sel']:>5.1f}% of selected)")
    lines.append(f"    Background: {stats['n_bg_sel']:>8,}  ({stats['bg_pct_sel']:>5.1f}% of selected)")
    lines.append("")

    # Edges
    lines.append(f"  Edges: E_total = {stats['E_total']:,}  (intra: {stats['E_intra']:,}, cross-layer: {stats['E_cross']:,})")
    lines.append(f"    Edges per node: {stats['edges_per_node']:.2f}")
    if stats.get('is_compressed', False):
        lines.append(f"    [COMPRESSED] canonical(i<j): {stats['n_canonical']:,}, unique: {stats['n_unique_after_canon']:,}")
    approx_full_edges = stats['full_grid_edges_6nn']
    edge_compression = stats['E_total'] / max(approx_full_edges, 1)
    lines.append(f"    Approx full-grid 6-nn edges: ~{approx_full_edges:,}")
    lines.append(f"    Edge compression ratio: {edge_compression:.4f} = {edge_compression*100:.1f}%")
    lines.append("")

    # Edge breakdown
    lines.append(f"  Edge Breakdown by Zone:")
    lines.append(f"    Core-related:        {stats['n_core_edges']:>8,}")
    lines.append(f"    Ring-related:        {stats['n_ring_edges']:>8,}")
    lines.append(f"    Background-only:     {stats['n_bg_edges']:>8,}")
    lines.append(f"    Backbone-BG:         {stats['n_bb_bg_edges']:>8,}")
    lines.append("")

    # Backbone in active zones
    lines.append(f"  Backbone in Active Zones:")
    lines.append(f"    In Core:  {stats['bb_in_core']:>6} / {stats['N_backbone']:,}")
    lines.append(f"    In Ring:  {stats['bb_in_ring']:>6} / {stats['N_backbone']:,}")
    lines.append(f"    In BG:    {stats['bb_in_bg']:>6} / {stats['N_backbone']:,}")
    lines.append(f"    Total active BB: {stats['bb_in_active']} ({stats['bb_in_active_pct']:.1f}%)")
    lines.append("")

    # Connectivity
    lines.append(f"  Per-Zone Average Degree (connectivity):")
    lines.append(f"    Core: {stats['avg_core_degree']:.2f}    Ring: {stats['avg_ring_degree']:.2f}    BG: {stats['avg_bg_degree']:.2f}")
    lines.append(f"  Per-Zone Edge Density (edges / zone_nodes):")
    lines.append(f"    Core: {stats['core_edge_density']:.2f}    Ring: {stats['ring_edge_density']:.2f}    BG: {stats['bg_edge_density']:.2f}")
    lines.append("")

    # Active-first verdict
    lines.append(f"  Active-First Effectiveness:")
    core_vs_bg = stats['avg_core_degree'] / max(stats['avg_bg_degree'], 0.01)
    ring_vs_bg = stats['avg_ring_degree'] / max(stats['avg_bg_degree'], 0.01)
    if core_vs_bg > 2.0:
        verdict_core = f"GOOD (core/bg = {core_vs_bg:.1f}x)"
    elif core_vs_bg > 1.0:
        verdict_core = f"WEAK (core/bg = {core_vs_bg:.1f}x, should be >2x)"
    else:
        verdict_core = f"BAD  (core/bg = {core_vs_bg:.1f}x, no advantage!)"

    if ring_vs_bg > 1.5:
        verdict_ring = f"GOOD (ring/bg = {ring_vs_bg:.1f}x)"
    elif ring_vs_bg > 1.0:
        verdict_ring = f"WEAK (ring/bg = {ring_vs_bg:.1f}x)"
    else:
        verdict_ring = f"BAD  (ring/bg = {ring_vs_bg:.1f}x)"

    lines.append(f"    Core connectivity vs BG: {verdict_core}")
    lines.append(f"    Ring connectivity vs BG: {verdict_ring}")
    lines.append("")

    # Score stats
    lines.append(f"  Activity Score Stats:")
    lines.append(f"    Overall mean={stats['score_mean']:.4f}, std={stats['score_std']:.4f}")
    lines.append(f"    Core mean={stats['score_core_mean']:.4f}, Ring mean={stats['score_ring_mean']:.4f}, BG mean={stats['score_bg_mean']:.4f}")

    return "\n".join(lines)


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnose adaptive graph effectiveness")
    parser.add_argument("--config", type=str, default="config/adaptive_graph_new.json")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of data samples to diagnose")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon (default: from config)")
    parser.add_argument("--epoch", type=int, default=0, help="Simulated epoch for gt_blend decay")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args()

    # Load config
    args = load_json_config(cli.config)
    if cli.seed is not None:
        set_seed(cli.seed)

    device_str = cli.device or getattr(args, "device", "cuda:0")
    if "cuda" in device_str and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    data_cfg = args.data
    adaptive_cfg = getattr(args, "adaptive_graph", None) or {}
    horizon = cli.horizon or data_cfg.get("horizon_train", 5)

    print("=" * 78)
    print(f"  Adaptive Graph Diagnostic  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: {cli.config}")
    print(f"  Device: {device_str}  |  Epoch (simulated): {cli.epoch}  |  Horizon: {horizon}")
    print("=" * 78)
    print()

    # Print config summary
    print("[Config Summary]")
    bb_cfg = adaptive_cfg.get("backbone", {})
    az_cfg = adaptive_cfg.get("active_zone", {})
    ref_cfg = adaptive_cfg.get("refinement", {})
    edge_cfg = adaptive_cfg.get("edges", {})
    wb_cfg = adaptive_cfg.get("writeback", {})

    print(f"  Backbone stride: {bb_cfg.get('stride', [2,2,2])}")
    print(f"  Refresh K: {ref_cfg.get('refresh_every_K', 4)}")
    print(f"  GT blend: {ref_cfg.get('gt_blend_start', 1.0)} -> {ref_cfg.get('gt_blend_end', 0.0)} over {ref_cfg.get('gt_blend_warmup_epochs', 50)} epochs")
    print(f"  Score weights: grad={az_cfg.get('gradient_weight', 0.5)}, temp={az_cfg.get('temporal_weight', 0.3)}, phys={az_cfg.get('physics_weight', 0.2)}")
    print(f"  Thresholds: core>={az_cfg.get('core_threshold', 0.6)}, ring>={az_cfg.get('ring_threshold', 0.2)}")
    print(f"  Keep ratios: core={az_cfg.get('core_keep_ratio', 1.0)}, ring={az_cfg.get('ring_keep_ratio', 0.5)}, bg_extra={az_cfg.get('background_extra_ratio', 0.0)}")
    print(f"  Stencils: bg={edge_cfg.get('bg_stencil', 6)}, ring={edge_cfg.get('ring_stencil', 18)}, core={edge_cfg.get('core_stencil', 26)}")

    # Compression info
    comp_cfg = adaptive_cfg.get("compression", {})
    if comp_cfg.get("enabled", False):
        print(f"  COMPRESSION ENABLED:")
        print(f"    Core max neighbors: {comp_cfg.get('core_max_neighbors', 12)}")
        print(f"    Ring max neighbors: {comp_cfg.get('ring_max_neighbors', 8)}")
        print(f"    BG edge keep ratio: {comp_cfg.get('bg_edge_keep_ratio', 0.7)}")
        print(f"    Max edges ratio:    {comp_cfg.get('max_edges_ratio', 4.0)} * N_nodes")
        print(f"    BG backbone keep:   {az_cfg.get('bg_backbone_keep_ratio', 1.0)}")
    else:
        print(f"  Compression: DISABLED (full stencils)")

    print(f"  Writeback: method={wb_cfg.get('method', 'idw')}, delayed={wb_cfg.get('delayed', True)}, IDW power={wb_cfg.get('idw_power', 2.0)}, k={wb_cfg.get('idw_k_neighbors', 4)}")
    print()

    # Build dataset (reuse existing code)
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
    print(f"  Dataset: {len(dataset)} samples, node_num={dataset.node_num}, dt={dataset.dt:.4e}")
    print()

    # Diagnose n_samples
    timer = Timer()
    all_results = []
    n_samples = min(cli.n_samples, len(dataset))

    for sample_idx in range(n_samples):
        print(f"{'='*78}")
        print(f"  Sample {sample_idx + 1} / {n_samples}")
        print(f"{'='*78}")

        batch_raw = dataset[sample_idx]
        # add batch dimension
        batch = {}
        for k, v in batch_raw.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
            else:
                batch[k] = v

        with timer("total_sample"):
            results = diagnose_single_sample(
                batch=batch,
                adaptive_cfg=adaptive_cfg,
                device=device,
                timer=timer,
                epoch=cli.epoch,
                horizon=horizon,
                fields=data_cfg.get("fields", []),
            )

        for r in results:
            r["sample_idx"] = sample_idx
            print(format_stats(r))
            print()

        all_results.extend(results)

    # ---- Aggregate Summary ----
    print("=" * 78)
    print("  AGGREGATE SUMMARY")
    print("=" * 78)
    print()

    if all_results:
        n = len(all_results)
        agg_keys = [
            "N_full", "N_backbone", "N_selected",
            "n_core_full", "n_ring_full", "n_bg_full",
            "n_core_sel", "n_ring_sel", "n_bg_sel",
            "node_compression",
            "E_total", "E_intra", "E_cross",
            "edges_per_node",
            "n_core_edges", "n_ring_edges", "n_bg_edges",
            "avg_core_degree", "avg_ring_degree", "avg_bg_degree",
            "core_edge_density", "ring_edge_density", "bg_edge_density",
            "bb_in_active", "bb_in_active_pct",
            "score_mean", "score_std",
            "score_core_mean", "score_ring_mean", "score_bg_mean",
        ]
        print(f"  Over {n} refresh events from {n_samples} samples:")
        print()

        for key in agg_keys:
            vals = [r[key] for r in all_results]
            mean_v = np.mean(vals)
            std_v = np.std(vals) if len(vals) > 1 else 0.0
            if isinstance(vals[0], float) or mean_v < 100:
                print(f"    {key:<30} mean={mean_v:>12.4f}   std={std_v:>10.4f}")
            else:
                print(f"    {key:<30} mean={mean_v:>12,.0f}   std={std_v:>10,.0f}")

        print()

        # Overall verdicts
        avg_core_deg = np.mean([r["avg_core_degree"] for r in all_results])
        avg_ring_deg = np.mean([r["avg_ring_degree"] for r in all_results])
        avg_bg_deg = np.mean([r["avg_bg_degree"] for r in all_results])
        avg_compression = np.mean([r["node_compression"] for r in all_results])
        avg_bb_active_pct = np.mean([r["bb_in_active_pct"] for r in all_results])

        print("  VERDICTS:")
        print(f"    Node compression:          {avg_compression*100:.1f}% of full grid")
        print(f"    Backbone in active zones:  {avg_bb_active_pct:.1f}%")

        if avg_core_deg > 2 * avg_bg_deg:
            print(f"    Core connectivity:         GOOD ({avg_core_deg:.1f} vs BG {avg_bg_deg:.1f}, {avg_core_deg/max(avg_bg_deg,0.01):.1f}x)")
        else:
            print(f"    Core connectivity:         WEAK ({avg_core_deg:.1f} vs BG {avg_bg_deg:.1f}, {avg_core_deg/max(avg_bg_deg,0.01):.1f}x)")

        if avg_ring_deg > 1.5 * avg_bg_deg:
            print(f"    Ring connectivity:         GOOD ({avg_ring_deg:.1f} vs BG {avg_bg_deg:.1f}, {avg_ring_deg/max(avg_bg_deg,0.01):.1f}x)")
        else:
            print(f"    Ring connectivity:         WEAK ({avg_ring_deg:.1f} vs BG {avg_bg_deg:.1f}, {avg_ring_deg/max(avg_bg_deg,0.01):.1f}x)")

        avg_core_pct = np.mean([r["core_pct_full"] for r in all_results])
        avg_ring_pct = np.mean([r["ring_pct_full"] for r in all_results])
        if avg_core_pct < 1.0:
            print(f"    WARNING: Core zone is very small ({avg_core_pct:.2f}% of grid) - consider lowering core_threshold")
        if avg_core_pct + avg_ring_pct < 5.0:
            print(f"    WARNING: Active zones (core+ring) only {avg_core_pct+avg_ring_pct:.2f}% - consider lowering thresholds")
        if avg_compression > 0.5:
            print(f"    WARNING: Compression is low ({avg_compression*100:.0f}%) - not saving much compute")
        if avg_compression < 0.05:
            print(f"    WARNING: Compression is extreme ({avg_compression*100:.1f}%) - may lose too much information")

    # ---- Timing Summary ----
    print()
    print("=" * 78)
    print("  TIMING SUMMARY")
    print("=" * 78)
    print()
    print(timer.summary())

    # ---- Save results ----
    result_dir = os.path.join(str(PROJ_ROOT), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(result_dir, f"diag_adaptive_{timestamp}.txt")

    output_text = []
    output_text.append(f"Adaptive Graph Diagnostic Report")
    output_text.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_text.append(f"Config: {cli.config}")
    output_text.append(f"Device: {device_str}  Epoch: {cli.epoch}  Horizon: {horizon}")
    output_text.append(f"Samples: {n_samples}")
    output_text.append("")

    output_text.append("[Config]")
    output_text.append(json.dumps(adaptive_cfg, indent=2))
    output_text.append("")

    for r in all_results:
        output_text.append(format_stats(r))
        output_text.append("")

    if all_results:
        output_text.append("=" * 78)
        output_text.append("AGGREGATE")
        output_text.append("=" * 78)
        for key in agg_keys:
            vals = [r[key] for r in all_results]
            mean_v = np.mean(vals)
            output_text.append(f"  {key}: {mean_v:.6f}")

    output_text.append("")
    output_text.append("TIMING")
    output_text.append(timer.summary())

    with open(out_file, "w") as f:
        f.write("\n".join(output_text))
    print()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
