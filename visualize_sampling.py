#!/usr/bin/env python
"""Visualize adaptive graph sampling distribution.

Loads one data sample, runs zone scoring + sampling (with a trained
checkpoint or from scratch), and produces:
  1. 3D scatter of all full-resolution nodes, colored by zone
  2. 2D XY-slice at a chosen Z layer, showing zones + selected nodes
  3. 2D XY-slice of the physical field (T / alpha.air) with selected
     node overlay
  4. Zone & sampling statistics printed to console

Usage:
    python visualize_sampling.py [--config CONFIG] [--ckpt CKPT]
                                 [--time-idx T] [--z-layer Z]
"""

import json
import sys, os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---- project imports ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
from src.dataset_adaptive import _compute_downsample_indices


# =====================================================================
#  helpers
# =====================================================================

def load_sample(h5_path: str, fields: list, spatial_stride: list,
                backbone_stride: list, time_idx: int = 5,
                normalize: bool = True):
    """Load a single time-step from an HDF5 file, mimicking the dataset."""
    field_stats = {
        "T":              (5.2999e+02, 4.5454e+02),
        "Ux":             (4.0041e-05, 2.4173e-01),
        "Uy":             (-1.6900e-05, 2.5172e-01),
        "Uz":             (3.3602e-07, 1.1976e-01),
        "alpha.air":      (0, 1),
        "alpha.titanium": (0, 1),
        "gamma_liquid":   (0, 1),
    }
    with h5py.File(h5_path, "r") as f:
        block = f["mesh/block"][0].astype(int)
        grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)

        fullres_indices, fullres_ds_shape = _compute_downsample_indices(
            grid_shape, spatial_stride
        )
        point_all = f["point"][:]
        fullres_pos = torch.from_numpy(
            point_all[fullres_indices].astype(np.float32)
        )

        # load state at time_idx
        channels = []
        for fname in fields:
            d = f[f"state/{fname}"][time_idx][fullres_indices, 0]
            channels.append(d)
        state_raw = np.stack(channels, axis=-1).astype(np.float32)  # [N_full, C]
        state = torch.from_numpy(state_raw)

        # previous time step
        prev_idx = max(0, time_idx - 1)
        channels_prev = []
        for fname in fields:
            d = f[f"state/{fname}"][prev_idx][fullres_indices, 0]
            channels_prev.append(d)
        prev_state_raw = np.stack(channels_prev, axis=-1).astype(np.float32)
        prev_state = torch.from_numpy(prev_state_raw)

    # normalize
    if normalize:
        mean_list = [field_stats[fn][0] for fn in fields]
        std_list = [field_stats[fn][1] for fn in fields]
        mean_t = torch.tensor(mean_list, dtype=torch.float32)
        std_t = torch.tensor(std_list, dtype=torch.float32)
        state_norm = (state - mean_t) / std_t
        prev_norm = (prev_state - mean_t) / std_t
    else:
        state_norm = state
        prev_norm = prev_state

    # backbone
    bb_indices, bb_shape = build_backbone_indices(fullres_ds_shape, backbone_stride)
    bb_edges = build_backbone_edges(bb_shape)

    return {
        "fullres_pos": fullres_pos,               # [N_full, 3] raw
        "fullres_ds_shape": fullres_ds_shape,      # (nx, ny, nz)
        "state_norm": state_norm,                  # [N_full, C] normalized
        "prev_norm": prev_norm,                    # [N_full, C] normalized
        "state_raw": torch.from_numpy(state_raw),  # [N_full, C] raw
        "backbone_indices": bb_indices,
        "backbone_shape": bb_shape,
        "backbone_edges": bb_edges,
    }


# =====================================================================
#  main
# =====================================================================

def main(
    config,
    time_idx: int | None = None,   # None → random
    z_layer: int | None = None,
    save: str = "./result/sample/sampling_viz.png"
):
    # ---- load config ----
    with open(config) as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    fields = data_cfg["fields"]
    spatial_stride = data_cfg.get("spatial_stride", [1, 1, 1])
    adaptive_graph_cfg = cfg.get("adaptive_graph", {})
    bb_stride = adaptive_graph_cfg.get("backbone", {}).get("stride", [2, 2, 2])

    # ---- pick an HDF5 file ----
    import random
    train_list = data_cfg["train_list"][0]
    with open(train_list) as f:
        h5_paths = [line.strip() for line in f if line.strip()]
    h5_path = random.choice(h5_paths)

    # ---- pick a time step ----
    if time_idx is None:
        with h5py.File(h5_path, "r") as f:
            first_field = data_cfg["fields"][0]
            n_steps = f[f"state/{first_field}"].shape[0]
        time_idx = random.randint(20, n_steps - 1)  # skip t=0 (no prev step)

    print(f"Loading: {h5_path}  (time_idx={time_idx})")

    # 更新图片保存路径
    os.makedirs("./result/sample", exist_ok=True)
    save = f"./result/sample/{os.path.basename(h5_path)}_{time_idx}_sample_viz.png"

    # ---- load data ----
    sample = load_sample(
        h5_path, fields, spatial_stride, bb_stride,
        time_idx=time_idx, normalize=True,
    )
    fullres_pos = sample["fullres_pos"]
    fr_shape = sample["fullres_ds_shape"]
    state_norm = sample["state_norm"]
    prev_norm = sample["prev_norm"]
    state_raw = sample["state_raw"]
    nx, ny, nz = fr_shape

    print(f"Full-res grid: {nx} x {ny} x {nz} = {nx*ny*nz} nodes")
    print(f"Fields: {fields}")

    # ---- build manager & run scoring ----
    device = torch.device("cpu")
    mgr = AdaptiveGraphManager(
        cfg=adaptive_graph_cfg,
        backbone_indices=sample["backbone_indices"],
        backbone_shape=sample["backbone_shape"],
        backbone_edges=sample["backbone_edges"],
        fullres_pos=fullres_pos,
        fullres_grid_shape=fr_shape,
        device=device,
        fields=fields,
    )

    az_cfg = adaptive_graph_cfg.get("active_zone", {})
    score = compute_active_score(
        pred_field=state_norm,
        gt_field=None,
        prev_field=prev_norm,
        grid_shape=fr_shape,
        gradient_weight=az_cfg.get("gradient_weight", 0.5),
        temporal_weight=az_cfg.get("temporal_weight", 0.3),
        physics_weight=az_cfg.get("physics_weight", 0.2),
        gt_blend=0.0,
        physics_triggers=mgr._physics_triggers,
    )

    zone = classify_zones(
        score,
        core_threshold=az_cfg.get("core_threshold", 0.6),
        ring_threshold=az_cfg.get("ring_threshold", 0.2),
    )

    selected = zone_first_sample(
        zone_full=zone,
        backbone_mask=mgr.backbone_mask,
        core_keep_ratio=az_cfg.get("core_keep_ratio", 1.0),
        ring_keep_ratio=az_cfg.get("ring_keep_ratio", 0.5),
        background_extra_ratio=az_cfg.get("background_extra_ratio", 0.0),
        bg_backbone_keep_ratio=az_cfg.get("bg_backbone_keep_ratio", 1.0),
    )

    # ---- build edges ----
    compress_cfg = adaptive_graph_cfg.get("compression", {})
    edge_cfg = adaptive_graph_cfg.get("edges", {})
    use_compressed = compress_cfg.get("enabled", False)

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

    full_to_local = torch.full((nx * ny * nz,), -1, dtype=torch.long)
    full_to_local[selected] = torch.arange(selected.shape[0])

    from src.adaptive_graph import build_cross_layer_edges
    cross_edges = build_cross_layer_edges(
        selected_indices=selected,
        zone_labels=zone,
        backbone_indices_set=mgr.bb_idx_tensor,
        backbone_stride=mgr.backbone_stride,
        grid_shape=fr_shape,
        full_to_local=full_to_local,
    )

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
        all_edges = torch.zeros((0, 2), dtype=torch.long)

    E_intra = intra_edges.shape[0]
    E_cross = cross_edges.shape[0] if cross_edges.numel() > 0 else 0
    E_before_budget = all_edges.shape[0]

    if use_compressed and all_edges.numel() > 0:
        sel_zone_labels = zone[selected]
        all_edges = enforce_edge_budget(
            edges=all_edges,
            zone_labels_selected=sel_zone_labels,
            max_edges_ratio=compress_cfg.get("max_edges_ratio", 4.0),
            num_nodes=selected.shape[0],
        )

    if use_compressed and all_edges.numel() > 0:
        sel_zone_labels = zone[selected]
        all_edges = enforce_per_node_degree(
            edges=all_edges,
            zone_labels_selected=sel_zone_labels,
            core_max_degree=compress_cfg.get("core_max_neighbors", 12),
            ring_max_degree=compress_cfg.get("ring_max_neighbors", 8),
            bg_max_degree=int(6 * compress_cfg.get("bg_edge_keep_ratio", 0.7)),
            num_nodes=selected.shape[0],
        )

    # ---- statistics ----
    n_core = (zone == 2).sum().item()
    n_ring = (zone == 1).sum().item()
    n_bg = (zone == 0).sum().item()
    n_sel = selected.shape[0]
    n_full = nx * ny * nz
    print(f"\n--- Zone Statistics ---")
    print(f"  Core:            {n_core:>7d}  ({100*n_core/n_full:.1f}%)")
    print(f"  Ring:            {n_ring:>7d}  ({100*n_ring/n_full:.1f}%)")
    print(f"  Background:      {n_bg:>7d}  ({100*n_bg/n_full:.1f}%)")
    print(f"  Selected:        {n_sel:>7d}  ({100*n_sel/n_full:.1f}%) out of {n_full}")

    # ---- edge statistics ----
    sel_zone = zone[selected]
    E_total = all_edges.shape[0]
    print(f"\n--- Edge Statistics ---")
    print(f"  Intra edges:     {E_intra:>7d}  (before budget)")
    print(f"  Cross edges:     {E_cross:>7d}  (before budget)")
    if use_compressed:
        print(f"  Before budget:   {E_before_budget:>7d}  ({E_before_budget/max(n_sel,1):.2f} per node)")
        print(f"  After budget:    {E_total:>7d}  ({E_total/max(n_sel,1):.2f} per node)  [max_ratio={compress_cfg.get('max_edges_ratio',4.0)}]")
    else:
        print(f"  Total edges:     {E_total:>7d}  ({E_total/max(n_sel,1):.2f} per node)")

    if E_total > 0:
        src_zone = sel_zone[all_edges[:, 0]]
        dst_zone = sel_zone[all_edges[:, 1]]
        n_core_edges = int(((src_zone == 2) | (dst_zone == 2)).sum().item())
        n_ring_edges = int(((src_zone == 1) | (dst_zone == 1)).sum().item())
        n_bg_edges   = int(((src_zone == 0) & (dst_zone == 0)).sum().item())
        print(f"  Core-related:    {n_core_edges:>7d}")
        print(f"  Ring-related:    {n_ring_edges:>7d}")
        print(f"  BG-only:         {n_bg_edges:>7d}")

        degree = torch.zeros(n_sel)
        flat_edges = all_edges.view(-1)
        degree.scatter_add_(0, flat_edges, torch.ones(flat_edges.shape[0]))
        avg_core_deg = float(degree[sel_zone == 2].mean().item()) if (sel_zone == 2).any() else 0.0
        avg_ring_deg = float(degree[sel_zone == 1].mean().item()) if (sel_zone == 1).any() else 0.0
        avg_bg_deg   = float(degree[sel_zone == 0].mean().item()) if (sel_zone == 0).any() else 0.0
        print(f"  Avg degree — Core: {avg_core_deg:.1f}  Ring: {avg_ring_deg:.1f}  BG: {avg_bg_deg:.1f}")

    # ---- prepare 3D coordinates ----
    # flat index → (x, y, z) in grid
    all_idx = torch.arange(n_full)
    gx = all_idx % nx
    gy = (all_idx // nx) % ny
    gz = all_idx // (nx * ny)

    sel_x = selected % nx
    sel_y = (selected // nx) % ny
    sel_z = selected // (nx * ny)

    # y is the height axis in this setup; slice along z for 2D panels
    z_layer = z_layer if z_layer is not None else nz // 2
    z_layer = min(z_layer, nz - 1)
    print(f"  Z-layer for 2D slice: {z_layer}")

    # ---- plot ----
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Adaptive Sampling Visualization — t={time_idx}  "
        f"grid={nx}x{ny}x{nz}  selected={n_sel}/{n_full} ({100*n_sel/n_full:.1f}%)",
        fontsize=13, fontweight="bold",
    )

    zone_colors = {0: "#d0d0d0", 1: "#4ea8de", 2: "#e63946"}
    zone_labels = {0: "Background", 1: "Ring", 2: "Core"}

    # ------------------------------------------------------------------
    # Panel 1: 3D scatter — all nodes colored by zone (subsampled for speed)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.set_title("3D Zone Distribution (subsampled)")

    # subsample for rendering speed
    max_pts_3d = 8000
    for z_val in [0, 1, 2]:
        mask = zone == z_val
        idx_z = torch.where(mask)[0]
        if idx_z.numel() == 0:
            continue
        if idx_z.numel() > max_pts_3d:
            perm = torch.randperm(idx_z.numel())[:max_pts_3d]
            idx_z = idx_z[perm]
        ax1.scatter(
            (idx_z % nx).numpy(),
            (idx_z // (nx * ny)).numpy(),
            ((idx_z // nx) % ny).numpy(),
            c=zone_colors[z_val], label=zone_labels[z_val],
            s=1 if z_val == 0 else 4, alpha=0.3 if z_val == 0 else 0.7,
        )
    ax1.set_xlabel("X"); ax1.set_ylabel("Z"); ax1.set_zlabel("Y (height)")
    ax1.legend(loc="upper left", fontsize=8, markerscale=5)

    # ------------------------------------------------------------------
    # Panel 2: 3D scatter — selected nodes only
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.set_title(f"3D Selected Nodes (N={n_sel})")

    max_pts_sel = 12000
    for z_val in [0, 1, 2]:
        mask_s = sel_zone == z_val
        idx_s = torch.where(mask_s)[0]
        if idx_s.numel() == 0:
            continue
        if idx_s.numel() > max_pts_sel:
            perm = torch.randperm(idx_s.numel())[:max_pts_sel]
            idx_s = idx_s[perm]
        ax2.scatter(
            sel_x[idx_s].numpy(), sel_z[idx_s].numpy(), sel_y[idx_s].numpy(),
            c=zone_colors[z_val], label=f"{zone_labels[z_val]} ({mask_s.sum().item()})",
            s=2, alpha=0.5 if z_val == 0 else 0.8,
        )
    ax2.set_xlabel("X"); ax2.set_ylabel("Z"); ax2.set_zlabel("Y (height)")
    ax2.legend(loc="upper left", fontsize=8, markerscale=5)

    # ------------------------------------------------------------------
    # Panel 3: 2D Z-slice — zone map
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title(f"Zone Map — Z-slice={z_layer}")

    # reshape zone to grid: [nz, ny, nx]
    zone_grid = zone.view(nz, ny, nx)
    zone_slice = zone_grid[z_layer].numpy()  # [ny, nx]

    cmap_zone = plt.cm.colors.ListedColormap(
        [zone_colors[0], zone_colors[1], zone_colors[2]]
    )
    ax3.imshow(zone_slice, origin="lower", cmap=cmap_zone, vmin=0, vmax=2,
               aspect="equal", interpolation="nearest")

    # overlay selected nodes at this z-layer
    sel_at_z = selected[sel_z == z_layer]
    sel_at_z_x = (sel_at_z % nx).numpy()
    sel_at_z_y = ((sel_at_z // nx) % ny).numpy()
    ax3.scatter(sel_at_z_x, sel_at_z_y, c="black", s=1, alpha=0.6,
                label=f"Selected at z={z_layer}")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y (height)")
    ax3.legend(fontsize=8, markerscale=5)

    # ------------------------------------------------------------------
    # Panel 4: 2D Z-slice — Temperature field + selected overlay
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    t_idx = fields.index("T") if "T" in fields else 0
    field_name = fields[t_idx]
    ax4.set_title(f"{field_name} Field — Z-slice={z_layer}")

    field_grid = state_raw[:, t_idx].view(nz, ny, nx)
    field_slice = field_grid[z_layer].numpy()  # [ny, nx]
    im4 = ax4.imshow(field_slice, origin="lower", cmap="hot", aspect="equal")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label=field_name)
    ax4.scatter(sel_at_z_x, sel_at_z_y, c="cyan", s=1, alpha=0.4,
                label="Selected")
    ax4.set_xlabel("X"); ax4.set_ylabel("Y (height)")
    ax4.legend(fontsize=8, markerscale=5)

    # ------------------------------------------------------------------
    # Panel 5: 2D Z-slice — second field (alpha.air if available)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)
    f2_idx = 1 if len(fields) > 1 else 0
    f2_name = fields[f2_idx]
    ax5.set_title(f"{f2_name} Field — Z-slice={z_layer}")

    f2_grid = state_raw[:, f2_idx].view(nz, ny, nx)
    f2_slice = f2_grid[z_layer].numpy()  # [ny, nx]
    im5 = ax5.imshow(f2_slice, origin="lower", cmap="coolwarm", aspect="equal")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label=f2_name)
    ax5.scatter(sel_at_z_x, sel_at_z_y, c="lime", s=1, alpha=0.4,
                label="Selected")
    ax5.set_xlabel("X"); ax5.set_ylabel("Y (height)")
    ax5.legend(fontsize=8, markerscale=5)

    # ------------------------------------------------------------------
    # Panel 6: Activity score heatmap
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title(f"Activity Score — Z-slice={z_layer}")

    score_grid = score.view(nz, ny, nx)
    score_slice = score_grid[z_layer].numpy()  # [ny, nx]
    im6 = ax6.imshow(score_slice, origin="lower", cmap="inferno", aspect="equal",
                     vmin=0, vmax=1)
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label="Score")

    # draw zone boundaries
    from matplotlib.colors import ListedColormap
    zone_boundary = np.zeros_like(zone_slice, dtype=float)
    zone_boundary[zone_slice == 1] = 0.5
    zone_boundary[zone_slice == 2] = 1.0
    ax6.contour(zone_boundary, levels=[0.25, 0.75], colors=["#4ea8de", "#e63946"],
                linewidths=1.0, linestyles="--")
    ax6.set_xlabel("X"); ax6.set_ylabel("Y (height)")

    plt.tight_layout()
    plt.savefig(save, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {save}")
    plt.close(fig)


if __name__ == "__main__":
    main(config="config/adaptive_graph_compressed.json",)
