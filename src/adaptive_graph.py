"""
adaptive_graph.py  (zone-first refactored)
==========================================
Dynamic adaptive graph construction for LPBF physics field prediction.

## Core design principle (本次重构核心):
    "先分区, 再采样, 再建边"
    Zone-first, then sample, then build edges.

    backbone 现在只是背景参考层 / 全局粗网格存档层。
    active 区域优先级高于 backbone。
    所有采样和建边决策以 zone 为主, 而不是以 backbone 身份为主。

Architecture:
  1. Active zone scoring   – regular-grid finite-difference gradient + temporal
                             change + physics-trigger terms (Section 1)
  2. Zone classification   – core / ring / background (Section 2)
  3. Zone-first sampling   – zone-priority, not backbone-priority (Section 3)
  4. Stencil edge builder  – structured grid stencils per zone (Section 4)
  5. Cross-layer edges     – coarse-fine parent-child mapping (Section 5)
  6. Graph refresh + delayed writeback manager (Section 6)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ===================================================================
#  1. Active zone scoring  (规则网格有限差分 + 时间变化 + 物理触发)
# ===================================================================

def compute_active_score(
    pred_field: torch.Tensor,
    gt_field: Optional[torch.Tensor],
    prev_field: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    gradient_weight: float = 0.5,
    temporal_weight: float = 0.3,
    physics_weight: float = 0.2,
    gt_blend: float = 1.0,
    physics_triggers: Optional[Dict] = None,
) -> torch.Tensor:
    """Per-node activity score in [0, 1].

    Composed of three terms (all normalized to [0,1] before blending):
      a) spatial gradient   regular-grid 6-neighbor finite difference
      b) temporal change    |field_t - field_{t-1}|
      c) physics triggers   per-channel thresholding (e.g. high T, velocity mag)

    Uses regular cubic grid finite-difference instead of KNN gradient,
    exploiting the structured-grid nature of LPBF data.

    Args:
        pred_field:  [N, C] predicted state (normalized).
        gt_field:    [N, C] ground-truth state (normalized), or None.
        prev_field:  [N, C] previous-step state (normalized).
        grid_shape:  (nx, ny, nz) of the full-resolution structured grid.
        physics_triggers: dict mapping channel_index -> threshold, e.g.
            {0: 0.8, 3: 0.5}.  A node scores 1 for a channel if its
            normalized value exceeds the threshold.
    """
    if gt_field is not None and gt_blend > 0.0:
        field = gt_blend * gt_field + (1.0 - gt_blend) * pred_field
    else:
        field = pred_field

    device = field.device

    # --- a) spatial gradient via regular-grid finite difference ---
    spatial_grad = _grid_finite_difference_gradient(field, grid_shape)

    # --- b) temporal change rate ---
    temporal_change = (field - prev_field).abs().mean(dim=-1)

    # --- c) physics-trigger score ---
    physics_score = _physics_trigger_score(field, physics_triggers)

    # normalize each to [0, 1]
    spatial_grad = _safe_minmax(spatial_grad)
    temporal_change = _safe_minmax(temporal_change)
    # physics_score is already in [0, 1]

    score = (gradient_weight * spatial_grad
             + temporal_weight * temporal_change
             + physics_weight * physics_score)
    
    return score.clamp(0.0, 1.0)


def _grid_finite_difference_gradient(
    field: torch.Tensor,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Approximate spatial gradient magnitude using 6-neighbor stencil on
    the regular cubic grid (central differences).

    Much faster than KNN-based gradient: O(N) with no distance computation.

    Returns:
        grad_mag: [N] per-node gradient magnitude.
    """
    nx, ny, nz = grid_shape
    C = field.shape[-1]
    # reshape to (nz, ny, nx, C) matching the flat-index layout:
    #   flat = z * nx * ny + y * nx + x
    vol = field.view(nz, ny, nx, C)

    grad_sq = torch.zeros(nz, ny, nx, device=field.device)

    # central differences along each axis, forward/backward at boundaries
    # x-axis (dim=2)
    if nx > 1:
        dx = torch.zeros_like(vol)
        dx[:, :, 1:-1] = (vol[:, :, 2:] - vol[:, :, :-2]) / 2.0
        dx[:, :, 0] = vol[:, :, 1] - vol[:, :, 0]
        dx[:, :, -1] = vol[:, :, -1] - vol[:, :, -2]
        grad_sq += (dx ** 2).sum(dim=-1)

    # y-axis (dim=1)
    if ny > 1:
        dy = torch.zeros_like(vol)
        dy[:, 1:-1] = (vol[:, 2:] - vol[:, :-2]) / 2.0
        dy[:, 0] = vol[:, 1] - vol[:, 0]
        dy[:, -1] = vol[:, -1] - vol[:, -2]
        grad_sq += (dy ** 2).sum(dim=-1)

    # z-axis (dim=0)
    if nz > 1:
        dz = torch.zeros_like(vol)
        dz[1:-1] = (vol[2:] - vol[:-2]) / 2.0
        dz[0] = vol[1] - vol[0]
        dz[-1] = vol[-1] - vol[-2]
        grad_sq += (dz ** 2).sum(dim=-1)

    grad_mag = grad_sq.sqrt().view(-1)  # [N]
    return grad_mag


def _physics_trigger_score(
    field: torch.Tensor,
    triggers: Optional[Dict],
) -> torch.Tensor:
    """Per-node physics-based trigger score in [0, 1].

    Each trigger maps a channel index to a threshold.  If the node's
    normalized value for that channel exceeds the threshold, that
    channel contributes 1.0; otherwise 0.0.  The final score is
    the max across triggered channels (any-of semantics).

    If no triggers are configured, returns zeros (neutral contribution).
    """
    N = field.shape[0]
    device = field.device
    if not triggers:
        return torch.zeros(N, device=device)

    score = torch.zeros(N, device=device)
    for ch_idx, threshold in triggers.items():
        ch_idx = int(ch_idx)
        if ch_idx < field.shape[-1]:
            # use abs value so both positive/negative extremes trigger
            ch_val = field[:, ch_idx].abs()
            ch_norm = _safe_minmax(ch_val)
            triggered = (ch_norm >= threshold).float()
            score = torch.max(score, triggered)
    return score


def _safe_minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize to [0, 1]; returns zeros if range is negligible."""
    xmin, xmax = x.min(), x.max()
    rng = xmax - xmin
    if rng < 1e-12:
        return torch.zeros_like(x)
    return (x - xmin) / rng


# ===================================================================
#  2. Zone classification (core / ring / background)
# ===================================================================

def classify_zones(
    score: torch.Tensor,
    core_threshold: float = 0.6,
    ring_threshold: float = 0.2,
) -> torch.Tensor:
    """Classify nodes: 2=core, 1=ring, 0=background.

    This operates on the FULL grid — backbone identity is irrelevant here.
    """
    zone = torch.zeros_like(score, dtype=torch.long)
    zone[score >= ring_threshold] = 1
    zone[score >= core_threshold] = 2
    return zone


# ===================================================================
#  3. Zone-first point sampling
#     核心原则: 先按 zone 决定保留, 而不是先按 backbone 身份决定保留.
#     backbone 点如果落在 core/ring 里, 按 core/ring 规则保留.
#     background 区域只保留 coarse backbone 点.
# ===================================================================

def zone_first_sample(
    zone_full: torch.Tensor,
    backbone_mask: torch.Tensor,
    core_keep_ratio: float = 1.0,
    ring_keep_ratio: float = 0.5,
    background_extra_ratio: float = 0.0,
    bg_backbone_keep_ratio: float = 1.0,
) -> torch.Tensor:
    """Zone-priority sampling — zone determines keep ratio, not backbone status.

    Logic:
      - core (zone==2):  keep core_keep_ratio of ALL core nodes (backbone or not)
      - ring  (zone==1):  keep ring_keep_ratio of ALL ring nodes
      - background (zone==0): keep backbone nodes (subsampled by bg_backbone_keep_ratio)
        optionally keep background_extra_ratio of non-backbone background nodes

    Returns:
        selected_indices: 1-D long tensor, indices into full grid for the
            combined sub-graph (includes backbone nodes in active zones AND
            non-backbone active nodes).
    """
    device = zone_full.device
    selected = []

    # ---- core zone (highest priority) ----
    core_all = torch.where(zone_full == 2)[0]
    if core_all.numel() > 0:
        n_keep = max(1, int(core_all.numel() * core_keep_ratio))
        if n_keep < core_all.numel():
            perm = torch.randperm(core_all.numel(), device=device)[:n_keep]
            selected.append(core_all[perm])
        else:
            selected.append(core_all)

    # ---- ring zone ----
    ring_all = torch.where(zone_full == 1)[0]
    if ring_all.numel() > 0:
        n_keep = max(1, int(ring_all.numel() * ring_keep_ratio))
        if n_keep < ring_all.numel():
            perm = torch.randperm(ring_all.numel(), device=device)[:n_keep]
            selected.append(ring_all[perm])
        else:
            selected.append(ring_all)

    # ---- background zone: backbone nodes (with optional subsampling) ----
    bg_backbone = torch.where((zone_full == 0) & backbone_mask)[0]
    if bg_backbone.numel() > 0:
        if bg_backbone_keep_ratio < 1.0:
            n_keep = max(1, int(bg_backbone.numel() * bg_backbone_keep_ratio))
            perm = torch.randperm(bg_backbone.numel(), device=device)[:n_keep]
            selected.append(bg_backbone[perm])
        else:
            selected.append(bg_backbone)

    # optional: small fraction of non-backbone background for smoothness
    if background_extra_ratio > 0.0:
        bg_non_bb = torch.where((zone_full == 0) & (~backbone_mask))[0]
        if bg_non_bb.numel() > 0:
            n_keep = max(1, int(bg_non_bb.numel() * background_extra_ratio))
            perm = torch.randperm(bg_non_bb.numel(), device=device)[:n_keep]
            selected.append(bg_non_bb[perm])

    if selected:
        all_selected = torch.cat(selected)
        # deduplicate (a backbone node in core was already selected by core)
        all_selected = torch.unique(all_selected)
        return all_selected

    # fallback: at least backbone
    return torch.where(backbone_mask)[0]


# ===================================================================
#  4. Stencil-based edge construction on the regular grid
#     利用规则立方体网格优势, 用结构化 stencil 替代 KNN/radius graph.
# ===================================================================

# Pre-computed stencil offsets (dx, dy, dz) for 6/18/26 connectivity
_STENCIL_6 = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
]

_STENCIL_18 = _STENCIL_6 + [
    (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
    (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
    (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
]

_STENCIL_26 = _STENCIL_18 + [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
]


def build_stencil_edges(
    selected_indices: torch.Tensor,
    zone_labels: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    backbone_mask: torch.Tensor,
    bg_stencil: int = 6,
    ring_stencil: int = 18,
    core_stencil: int = 26,
) -> torch.Tensor:
    """Build edges using structured-grid stencils, per-zone connectivity.

    Each selected node gets edges to neighbors that are also selected,
    with the stencil size determined by the node's zone:
      - background/backbone: 6-neighbor
      - ring:                18-neighbor (or 6)
      - core:                26-neighbor (or 18)

    This replaces the old KNN/radius-based refinement edge construction.

    Args:
        selected_indices: [N_sel] indices into the full grid (flat).
        zone_labels:      [N_full] zone labels for the full grid.
        grid_shape:       (nx, ny, nz).
        backbone_mask:    [N_full] bool.
        bg_stencil:       stencil size for background (6).
        ring_stencil:     stencil size for ring (6 or 18).
        core_stencil:     stencil size for core (18 or 26).

    Returns:
        edges: [E, 2] long tensor in LOCAL indexing (0-based within selected_indices).
    """
    return _build_stencil_edges_impl(
        selected_indices, zone_labels, grid_shape, backbone_mask,
        bg_stencil, ring_stencil, core_stencil,
        compressed=False,
    )


def build_stencil_edges_compressed(
    selected_indices: torch.Tensor,
    zone_labels: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    backbone_mask: torch.Tensor,
    core_max_neighbors: int = 12,
    ring_max_neighbors: int = 8,
    bg_keep_ratio: float = 0.7,
) -> torch.Tensor:
    """Compressed stencil edge builder with per-zone neighbor budgets.

    Edge strategy per zone:
      - core:  keep nearest 6 + random sample from remaining 26-neighbors
               up to core_max_neighbors total (~10-12)
      - ring:  keep nearest 6 + extend up to ring_max_neighbors (~8-10)
      - background: keep 6-neighbors, randomly drop (1 - bg_keep_ratio)

    All edges are canonicalized (i < j) and deduplicated.

    Returns:
        edges: [E, 2] long tensor in LOCAL indexing, canonical (src < dst).
    """
    return _build_stencil_edges_impl(
        selected_indices, zone_labels, grid_shape, backbone_mask,
        bg_stencil=6, ring_stencil=18, core_stencil=26,
        compressed=True,
        core_max_neighbors=core_max_neighbors,
        ring_max_neighbors=ring_max_neighbors,
        bg_keep_ratio=bg_keep_ratio,
    )


def _build_stencil_edges_impl(
    selected_indices: torch.Tensor,
    zone_labels: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    backbone_mask: torch.Tensor,
    bg_stencil: int = 6,
    ring_stencil: int = 18,
    core_stencil: int = 26,
    compressed: bool = False,
    core_max_neighbors: int = 12,
    ring_max_neighbors: int = 8,
    bg_keep_ratio: float = 0.7,
) -> torch.Tensor:
    """Shared implementation for both original and compressed stencil edges."""
    nx, ny, nz = grid_shape
    device = selected_indices.device
    N_sel = selected_indices.shape[0]

    if N_sel == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    stencil_map = {6: _STENCIL_6, 18: _STENCIL_18, 26: _STENCIL_26}
    bg_offsets = stencil_map.get(bg_stencil, _STENCIL_6)
    ring_offsets = stencil_map.get(ring_stencil, _STENCIL_18)
    core_offsets = stencil_map.get(core_stencil, _STENCIL_26)

    N_full = nx * ny * nz
    full_to_local = torch.full((N_full,), -1, dtype=torch.long, device=device)
    full_to_local[selected_indices] = torch.arange(N_sel, device=device)

    sel_x = selected_indices % nx
    sel_y = (selected_indices // nx) % ny
    sel_z = selected_indices // (nx * ny)

    sel_zone = zone_labels[selected_indices]

    def _offsets_tensor(offsets_list):
        return torch.tensor(offsets_list, dtype=torch.long, device=device)

    bg_off_t = _offsets_tensor(bg_offsets)
    ring_off_t = _offsets_tensor(ring_offsets)
    core_off_t = _offsets_tensor(core_offsets)

    # For compressed mode: 6-neighbor offsets (faces only) as the guaranteed base
    base6_off_t = _offsets_tensor(_STENCIL_6)

    edge_list = []

    for zone_val, off_t in [(0, bg_off_t), (1, ring_off_t), (2, core_off_t)]:
        mask = (sel_zone == zone_val)
        if not mask.any():
            continue

        local_ids = torch.where(mask)[0]
        n_zone = local_ids.shape[0]
        K = off_t.shape[0]

        zx = sel_x[local_ids]
        zy = sel_y[local_ids]
        zz = sel_z[local_ids]

        nb_x = zx.unsqueeze(1) + off_t[:, 0].unsqueeze(0)
        nb_y = zy.unsqueeze(1) + off_t[:, 1].unsqueeze(0)
        nb_z = zz.unsqueeze(1) + off_t[:, 2].unsqueeze(0)

        valid = ((nb_x >= 0) & (nb_x < nx) &
                 (nb_y >= 0) & (nb_y < ny) &
                 (nb_z >= 0) & (nb_z < nz))

        nb_flat = nb_z * (nx * ny) + nb_y * nx + nb_x
        nb_flat = nb_flat.clamp(0, N_full - 1)

        nb_local = full_to_local[nb_flat]
        in_selected = (nb_local >= 0) & valid

        if compressed:
            # --- Compressed mode: per-zone neighbor budget ---
            if zone_val == 2:  # core
                # Keep base 6-neighbors unconditionally, sample extras up to budget
                in_selected = _budget_sample_neighbors(
                    local_ids, sel_x, sel_y, sel_z, off_t, base6_off_t,
                    in_selected, nb_local, core_max_neighbors, nx, ny, nz,
                    N_full, full_to_local, device,
                )
            elif zone_val == 1:  # ring
                in_selected = _budget_sample_neighbors(
                    local_ids, sel_x, sel_y, sel_z, off_t, base6_off_t,
                    in_selected, nb_local, ring_max_neighbors, nx, ny, nz,
                    N_full, full_to_local, device,
                )
            elif zone_val == 0:  # background
                # Random drop from 6-neighbors
                if bg_keep_ratio < 1.0 and in_selected.any():
                    drop_mask = torch.rand(in_selected.shape, device=device) > bg_keep_ratio
                    in_selected = in_selected & (~drop_mask)

        src_local = local_ids.unsqueeze(1).expand(-1, K)

        src_edges = src_local[in_selected]
        dst_edges = nb_local[in_selected]

        if src_edges.numel() > 0:
            edge_list.append(torch.stack([src_edges, dst_edges], dim=1))

    if edge_list:
        edges = torch.cat(edge_list, dim=0)
        # remove self-loops
        mask = edges[:, 0] != edges[:, 1]
        edges = edges[mask]

        if compressed:
            # canonicalize and deduplicate
            edges = canonicalize_and_dedup_edges(edges)

        return edges

    return torch.zeros((0, 2), dtype=torch.long, device=device)


def _budget_sample_neighbors(
    local_ids: torch.Tensor,
    sel_x: torch.Tensor,
    sel_y: torch.Tensor,
    sel_z: torch.Tensor,
    full_off_t: torch.Tensor,
    base6_off_t: torch.Tensor,
    in_selected: torch.Tensor,
    nb_local: torch.Tensor,
    max_neighbors: int,
    nx: int, ny: int, nz: int,
    N_full: int,
    full_to_local: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply neighbor budget: keep all base-6 neighbors, sample from extras.

    For each node, guarantees its 6-face-neighbors (if they exist in the
    selected set), then randomly samples from the remaining stencil neighbors
    up to max_neighbors total.

    Returns:
        Modified in_selected mask [n_zone, K].
    """
    n_zone = local_ids.shape[0]
    K_full = full_off_t.shape[0]

    # Identify which of the K_full offsets are base-6
    # (match by offset value)
    is_base6 = torch.zeros(K_full, dtype=torch.bool, device=device)
    for i in range(K_full):
        for j in range(base6_off_t.shape[0]):
            if (full_off_t[i] == base6_off_t[j]).all():
                is_base6[i] = True
                break

    # base6 mask: [n_zone, K_full]
    base6_mask = is_base6.unsqueeze(0).expand(n_zone, -1)
    extra_mask = ~base6_mask

    # base-6 edges (always keep)
    base_edges = in_selected & base6_mask

    # count base-6 per node
    base_count = base_edges.sum(dim=1)  # [n_zone]

    # extra candidates
    extra_candidates = in_selected & extra_mask

    # budget remaining per node
    budget = (max_neighbors - base_count).clamp(min=0)  # [n_zone]

    # For each node, randomly sample from extras up to budget
    extra_count = extra_candidates.sum(dim=1)  # [n_zone]
    needs_trim = extra_count > budget

    if needs_trim.any():
        # Vectorized random sampling: assign random priorities, keep top-budget
        rand_scores = torch.rand(n_zone, K_full, device=device)
        rand_scores[~extra_candidates] = -1.0  # ensure non-candidates rank lowest

        # For nodes that need trimming, zero out lowest-priority extras
        # Sort by random score per row, keep top-budget
        trim_ids = torch.where(needs_trim)[0]
        for idx in trim_ids:
            b = int(budget[idx].item())
            if b <= 0:
                extra_candidates[idx] = False
                continue
            row_scores = rand_scores[idx]
            row_cands = extra_candidates[idx]
            cand_indices = torch.where(row_cands)[0]
            if cand_indices.numel() > b:
                cand_scores = row_scores[cand_indices]
                _, topk_local = cand_scores.topk(b)
                keep_set = cand_indices[topk_local]
                new_row = torch.zeros(K_full, dtype=torch.bool, device=device)
                new_row[keep_set] = True
                extra_candidates[idx] = new_row

    # Combine base + trimmed extras
    result = base_edges | extra_candidates
    return result


def canonicalize_and_dedup_edges(edges: torch.Tensor) -> torch.Tensor:
    """Canonicalize edges to (min, max) ordering and remove duplicates.

    Converts all edges to undirected (i < j) and removes duplicates.

    Args:
        edges: [E, 2] long tensor.

    Returns:
        deduped: [E', 2] long tensor with E' <= E, all edges have src < dst.
    """
    if edges.numel() == 0:
        return edges

    # Canonicalize: ensure src < dst
    src = edges[:, 0]
    dst = edges[:, 1]
    canon_src = torch.min(src, dst)
    canon_dst = torch.max(src, dst)
    canon = torch.stack([canon_src, canon_dst], dim=1)

    # Deduplicate using unique
    canon_deduped = torch.unique(canon, dim=0)
    return canon_deduped


def enforce_edge_budget(
    edges: torch.Tensor,
    zone_labels_selected: torch.Tensor,
    max_edges_ratio: float = 4.0,
    num_nodes: int = 0,
) -> torch.Tensor:
    """Enforce a global edge budget: max_edges ≈ max_edges_ratio * num_nodes.

    Priority deletion order: background edges first, then ring, then core.

    Args:
        edges:                [E, 2] canonical edges (src < dst).
        zone_labels_selected: [N_sel] zone labels for the selected nodes.
        max_edges_ratio:      max edges per node ratio.
        num_nodes:            number of selected nodes.

    Returns:
        trimmed: [E', 2] edges with E' <= max_edges.
    """
    if edges.numel() == 0:
        return edges

    max_edges = int(max_edges_ratio * max(num_nodes, 1))
    if edges.shape[0] <= max_edges:
        return edges

    # Classify each edge by zone: use max(zone_src, zone_dst)
    # Higher zone = more important
    src_zone = zone_labels_selected[edges[:, 0]]
    dst_zone = zone_labels_selected[edges[:, 1]]
    edge_zone = torch.max(src_zone, dst_zone)  # 0=bg, 1=ring, 2=core

    # Assign deletion priority: bg=0 (delete first), ring=1, core=2 (delete last)
    # Sort by priority ascending, keep first max_edges
    # Add random jitter within same priority for fair sampling
    priority = edge_zone.float() + torch.rand(edges.shape[0], device=edges.device) * 0.9
    _, keep_idx = priority.topk(max_edges, largest=True)
    keep_idx = keep_idx.sort().values  # maintain order

    return edges[keep_idx]


# ===================================================================
#  5. Cross-layer (coarse-fine) edges via parent-child grid mapping
#     用规则网格的 parent-child 映射替代全局 KNN 跨层连接.
# ===================================================================

def build_cross_layer_edges(
    selected_indices: torch.Tensor,
    zone_labels: torch.Tensor,
    backbone_indices_set: torch.Tensor,
    backbone_stride: Tuple[int, int, int],
    grid_shape: Tuple[int, int, int],
    full_to_local: torch.Tensor,
) -> torch.Tensor:
    """Build cross-layer edges between fine (core/ring) nodes and their
    coarse backbone parent cells using regular-grid parent-child mapping.

    For each fine node at (x,y,z), its parent backbone node is at
    (x // sx, y // sy, z // sz) * stride.  We connect the fine node to
    the nearest backbone node(s) using the stride relationship.

    This replaces the old KNN-based cross edges.

    Returns:
        edges: [E, 2] in local indexing (within selected_indices).
    """
    nx, ny, nz = grid_shape
    sx, sy, sz = backbone_stride
    device = selected_indices.device
    N_full = nx * ny * nz

    # fine nodes = core or ring nodes in the selected set
    sel_zone = zone_labels[selected_indices]
    fine_mask = (sel_zone >= 1)  # ring or core
    if not fine_mask.any():
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    fine_local = torch.where(fine_mask)[0]
    fine_full = selected_indices[fine_local]

    # convert to coordinates
    fine_x = fine_full % nx
    fine_y = (fine_full // nx) % ny
    fine_z = fine_full // (nx * ny)

    # parent backbone coordinates (nearest coarse grid node)
    # round to nearest stride multiple
    parent_x = ((fine_x + sx // 2) // sx * sx).clamp(0, nx - 1)
    parent_y = ((fine_y + sy // 2) // sy * sy).clamp(0, ny - 1)
    parent_z = ((fine_z + sz // 2) // sz * sz).clamp(0, nz - 1)

    parent_flat = parent_z * (nx * ny) + parent_y * nx + parent_x

    # check if parent is in our selected set
    parent_local = full_to_local[parent_flat]
    valid = (parent_local >= 0)

    # also avoid self-loops
    valid = valid & (fine_local != parent_local)

    if not valid.any():
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    # bidirectional edges
    src = fine_local[valid]
    dst = parent_local[valid]

    fwd = torch.stack([src, dst], dim=1)
    bwd = torch.stack([dst, src], dim=1)
    edges = torch.cat([fwd, bwd], dim=0)

    return edges


# ===================================================================
#  6. AdaptiveGraphManager  (zone-first, delayed writeback)
# ===================================================================

def build_backbone_indices(
    grid_shape: Tuple[int, int, int],
    backbone_stride: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Compute backbone (coarse grid) indices — kept for dataset compatibility.

    backbone 现在只是背景参考层 / 全局粗网格存档层, 不再是优先保留的主图.
    """
    gx, gy, gz = grid_shape
    sx, sy, sz = backbone_stride
    xs = list(range(0, gx, sx))
    ys = list(range(0, gy, sy))
    zs = list(range(0, gz, sz))
    if xs[-1] != gx - 1:
        xs.append(gx - 1)
    if ys[-1] != gy - 1:
        ys.append(gy - 1)
    if zs[-1] != gz - 1:
        zs.append(gz - 1)
    backbone_shape = (len(xs), len(ys), len(zs))
    indices = []
    for z in zs:
        for y in ys:
            base = z * gx * gy + y * gx
            for x in xs:
                indices.append(base + x)
    return np.asarray(indices, dtype=np.int64), backbone_shape


def build_backbone_edges(
    backbone_shape: Tuple[int, int, int],
    sample_ratio: float = 1.0,
) -> torch.Tensor:
    """6-neighbor edges for backbone — kept for dataset compatibility."""
    nx, ny, nz = backbone_shape
    idx_grid = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    parts = []
    if nx > 1:
        src = idx_grid[:, :, :-1].flatten()
        dst = idx_grid[:, :, 1:].flatten()
        parts.append(np.stack([src, dst], axis=1))
    if ny > 1:
        src = idx_grid[:, :-1, :].flatten()
        dst = idx_grid[:, 1:, :].flatten()
        parts.append(np.stack([src, dst], axis=1))
    if nz > 1:
        src = idx_grid[:-1, :, :].flatten()
        dst = idx_grid[1:, :, :].flatten()
        parts.append(np.stack([src, dst], axis=1))
    if parts:
        edges_arr = np.concatenate(parts, axis=0)
    else:
        edges_arr = np.zeros((0, 2), dtype=np.int64)
    if sample_ratio < 1.0 and edges_arr.shape[0] > 0:
        total = edges_arr.shape[0]
        keep = int(total * sample_ratio)
        sel = np.random.choice(total, keep, replace=False)
        edges_arr = edges_arr[sel]
    return torch.from_numpy(edges_arr.astype(np.int64))


class AdaptiveGraphManager:
    """Zone-first adaptive graph manager with delayed writeback.

    核心改动:
      1. backbone 只是背景参考层, active 区域优先级最高
      2. zone-first 采样: 先 compute_active_score → classify_zones →
         zone_first_sample, 而不是 "先保留 backbone 再补 refinement"
      3. stencil 建边: 用规则网格 stencil 替代 KNN/radius graph
      4. delayed writeback: 非 refresh 步不 materialize full grid,
         只在 refresh / full supervision / 最终输出步做回写
    """

    def __init__(
        self,
        cfg: dict,
        backbone_indices: np.ndarray,
        backbone_shape: Tuple[int, int, int],
        backbone_edges: torch.Tensor,
        fullres_pos: torch.Tensor,
        fullres_grid_shape: Tuple[int, int, int],
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device

        # backbone meta (背景参考层)
        self.backbone_indices_np = backbone_indices
        self.backbone_shape = backbone_shape
        self.n_backbone = len(backbone_indices)
        self.backbone_edges = backbone_edges.to(device)

        # full resolution meta
        self.fullres_pos = fullres_pos.to(device)
        self.n_full = fullres_pos.shape[0]
        self.fullres_grid_shape = fullres_grid_shape

        # backbone mask on full grid
        self.backbone_mask = torch.zeros(self.n_full, dtype=torch.bool, device=device)
        bb_idx_tensor = torch.from_numpy(backbone_indices).long().to(device)
        self.backbone_mask[bb_idx_tensor] = True
        self.bb_idx_tensor = bb_idx_tensor

        # backbone stride (needed for cross-layer edges)
        bb_cfg = cfg.get("backbone", {})
        self.backbone_stride = tuple(bb_cfg.get("stride", [2, 2, 2]))

        # --- current subgraph state ---
        self.selected_full_indices: Optional[torch.Tensor] = None  # [N_sel] in full grid
        self.selected_edges: Optional[torch.Tensor] = None         # [E, 2] local
        self.n_selected = 0
        self.zone_full: Optional[torch.Tensor] = None              # [N_full] zone labels

        # config shortcuts
        self.refresh_K = cfg.get("refinement", {}).get("refresh_every_K", 4)
        self.az_cfg = cfg.get("active_zone", {})
        self.edge_cfg = cfg.get("edges", {})
        self.wb_cfg = cfg.get("writeback", {})
        self.compress_cfg = cfg.get("compression", {})

        # physics trigger config: parse from active_zone config
        self._physics_triggers = self._parse_physics_triggers()

    def _parse_physics_triggers(self) -> Optional[Dict]:
        """Parse physics_triggers from config.

        Format in JSON: {"physics_triggers": {"0": 0.8, "1": 0.5}}
        channel_index -> threshold.
        """
        raw = self.az_cfg.get("physics_triggers", None)
        if raw and isinstance(raw, dict):
            return {int(k): float(v) for k, v in raw.items() if not k.startswith("_")}
        return None

    # ---- refresh logic -----------------------------------------------

    def should_refresh(self, t: int) -> bool:
        return t % self.refresh_K == 0

    def get_gt_blend(self, epoch: int) -> float:
        ref_cfg = self.cfg.get("refinement", {})
        start = ref_cfg.get("gt_blend_start", 1.0)
        end = ref_cfg.get("gt_blend_end", 0.0)
        warmup = ref_cfg.get("gt_blend_warmup_epochs", 30)
        if warmup <= 0:
            return end
        alpha = min(1.0, epoch / warmup)
        return start + alpha * (end - start)

    @torch.no_grad()
    def refresh(
        self,
        pred_field: torch.Tensor,
        gt_field: Optional[torch.Tensor],
        prev_field: torch.Tensor,
        epoch: int = 0,
    ):
        """Rebuild the subgraph: zone-first scoring → sampling → stencil edges.

        All tensors are [N_full, C] on device, in normalized space.
        Supports compressed mode via 'compression' config section.
        """
        gt_blend = self.get_gt_blend(epoch)
        use_compressed = self.compress_cfg.get("enabled", False)

        # Step 1: compute activity score (uses regular-grid finite difference)
        score = compute_active_score(
            pred_field=pred_field,
            gt_field=gt_field,
            prev_field=prev_field,
            grid_shape=self.fullres_grid_shape,
            gradient_weight=self.az_cfg.get("gradient_weight", 0.5),
            temporal_weight=self.az_cfg.get("temporal_weight", 0.3),
            physics_weight=self.az_cfg.get("physics_weight", 0.2),
            gt_blend=gt_blend,
            physics_triggers=self._physics_triggers,
        )

        # Step 2: zone classification
        zone = classify_zones(
            score,
            core_threshold=self.az_cfg.get("core_threshold", 0.6),
            ring_threshold=self.az_cfg.get("ring_threshold", 0.2),
        )
        self.zone_full = zone

        # Step 3: zone-first sampling (NOT backbone-first!)
        selected = zone_first_sample(
            zone_full=zone,
            backbone_mask=self.backbone_mask,
            core_keep_ratio=self.az_cfg.get("core_keep_ratio", 1.0),
            ring_keep_ratio=self.az_cfg.get("ring_keep_ratio", 0.5),
            background_extra_ratio=self.az_cfg.get("background_extra_ratio", 0.0),
            bg_backbone_keep_ratio=self.az_cfg.get("bg_backbone_keep_ratio", 1.0),
        )

        # Step 4: build edges
        if use_compressed:
            # Compressed stencil edges with per-zone neighbor budgets
            intra_edges = build_stencil_edges_compressed(
                selected_indices=selected,
                zone_labels=zone,
                grid_shape=self.fullres_grid_shape,
                backbone_mask=self.backbone_mask,
                core_max_neighbors=self.compress_cfg.get("core_max_neighbors", 12),
                ring_max_neighbors=self.compress_cfg.get("ring_max_neighbors", 8),
                bg_keep_ratio=self.compress_cfg.get("bg_edge_keep_ratio", 0.7),
            )
        else:
            intra_edges = build_stencil_edges(
                selected_indices=selected,
                zone_labels=zone,
                grid_shape=self.fullres_grid_shape,
                backbone_mask=self.backbone_mask,
                bg_stencil=self.edge_cfg.get("bg_stencil", 6),
                ring_stencil=self.edge_cfg.get("ring_stencil", 18),
                core_stencil=self.edge_cfg.get("core_stencil", 26),
            )

        # Step 5: cross-layer edges (fine→coarse parent-child mapping)
        N_full = self.n_full
        full_to_local = torch.full((N_full,), -1, dtype=torch.long, device=self.device)
        full_to_local[selected] = torch.arange(selected.shape[0], device=self.device)

        cross_edges = build_cross_layer_edges(
            selected_indices=selected,
            zone_labels=zone,
            backbone_indices_set=self.bb_idx_tensor,
            backbone_stride=self.backbone_stride,
            grid_shape=self.fullres_grid_shape,
            full_to_local=full_to_local,
        )

        # Combine intra + cross edges
        edge_parts = []
        if intra_edges.numel() > 0:
            edge_parts.append(intra_edges)
        if cross_edges.numel() > 0:
            if use_compressed:
                # Canonicalize cross edges too
                cross_edges = canonicalize_and_dedup_edges(cross_edges)
            edge_parts.append(cross_edges)

        if edge_parts:
            all_edges = torch.cat(edge_parts, dim=0)
            if use_compressed:
                # Final global dedup (intra already deduped, cross already deduped,
                # but there may be overlap between them)
                all_edges = canonicalize_and_dedup_edges(all_edges)
        else:
            all_edges = torch.zeros((0, 2), dtype=torch.long, device=self.device)

        # Step 6: enforce global edge budget (compressed mode only)
        if use_compressed and all_edges.numel() > 0:
            sel_zone = zone[selected]
            max_edges_ratio = self.compress_cfg.get("max_edges_ratio", 4.0)
            all_edges = enforce_edge_budget(
                edges=all_edges,
                zone_labels_selected=sel_zone,
                max_edges_ratio=max_edges_ratio,
                num_nodes=selected.shape[0],
            )

        # Step 7: remove isolated nodes (nodes with no edges)
        if all_edges.numel() > 0 and selected.shape[0] > 0:
            connected = torch.zeros(selected.shape[0], dtype=torch.bool, device=self.device)
            connected[all_edges[:, 0]] = True
            connected[all_edges[:, 1]] = True

            # always keep backbone nodes even if isolated (they're the reference layer)
            is_bb = self.backbone_mask[selected]
            connected = connected | is_bb

            if not connected.all():
                # remap
                keep_local = torch.where(connected)[0]
                new_local = torch.full((selected.shape[0],), -1, dtype=torch.long, device=self.device)
                new_local[keep_local] = torch.arange(keep_local.shape[0], device=self.device)

                selected = selected[keep_local]

                # remap edges
                valid_edges = (new_local[all_edges[:, 0]] >= 0) & (new_local[all_edges[:, 1]] >= 0)
                all_edges = all_edges[valid_edges]
                all_edges = torch.stack([new_local[all_edges[:, 0]], new_local[all_edges[:, 1]]], dim=1)

        self.selected_full_indices = selected
        self.n_selected = selected.shape[0]
        self.selected_edges = all_edges

    # ---- graph data access -------------------------------------------

    def get_selected_count(self) -> int:
        return self.n_selected

    def gather_state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract state for the selected subgraph from full-grid state.

        Args:
            full_state: [B, N_full, C] or [N_full, C].
        """
        idx = self.selected_full_indices
        if idx is None:
            idx = self.bb_idx_tensor
        if full_state.dim() == 3:
            return full_state[:, idx]
        return full_state[idx]

    def get_selected_pos(self) -> torch.Tensor:
        """Return [N_sel, 3] positions for the current subgraph."""
        idx = self.selected_full_indices
        if idx is None:
            idx = self.bb_idx_tensor
        return self.fullres_pos[idx]

    def get_selected_edges(self) -> torch.Tensor:
        """Return [E, 2] edges for the current subgraph."""
        if self.selected_edges is not None:
            return self.selected_edges
        return self.backbone_edges

    def get_selected_indices(self) -> torch.Tensor:
        """Return [N_sel] full-grid indices for the current subgraph."""
        if self.selected_full_indices is not None:
            return self.selected_full_indices
        return self.bb_idx_tensor

    # ---- writeback: subgraph prediction → full grid ------------------

    def writeback(
        self,
        subgraph_pred: torch.Tensor,
        full_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Write predictions from the subgraph back to the full grid.

        Direct copy for selected nodes; IDW interpolation for missing nodes.

        Args:
            subgraph_pred: [B, N_sel, C] predicted state on subgraph.
            full_prev:     [B, N_full, C] previous full-grid state.

        Returns:
            full_pred: [B, N_full, C].
        """
        B, _, C = subgraph_pred.shape
        full_pred = full_prev.clone()

        idx = self.get_selected_indices()
        full_pred[:, idx] = subgraph_pred

        # interpolate missing nodes
        all_indices = torch.arange(self.n_full, device=self.device)
        present_mask = torch.zeros(self.n_full, dtype=torch.bool, device=self.device)
        present_mask[idx] = True
        missing_indices = all_indices[~present_mask]

        if missing_indices.numel() == 0:
            return full_pred

        method = self.wb_cfg.get("method", "idw")
        if method == "idw":
            full_pred = self._idw_interpolate(
                full_pred, subgraph_pred, idx, missing_indices, B, C,
            )

        return full_pred

    def _idw_interpolate(
        self,
        full_pred: torch.Tensor,
        subgraph_pred: torch.Tensor,
        present_idx: torch.Tensor,
        missing_idx: torch.Tensor,
        B: int,
        C: int,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """IDW interpolation for missing nodes."""
        power = self.wb_cfg.get("idw_power", 2.0)
        k = self.wb_cfg.get("idw_k_neighbors", 4)

        selected_pos = self.get_selected_pos()
        missing_pos = self.fullres_pos[missing_idx]
        N_miss = missing_pos.shape[0]
        k_use = min(k, selected_pos.shape[0])

        for start in range(0, N_miss, chunk_size):
            end = min(start + chunk_size, N_miss)
            dists = torch.cdist(
                missing_pos[start:end].unsqueeze(0),
                selected_pos.unsqueeze(0),
            ).squeeze(0)

            _, nn_idx = dists.topk(k_use, dim=-1, largest=False)
            nn_dists = torch.gather(dists, 1, nn_idx)

            weights = 1.0 / (nn_dists.pow(power) + 1e-10)
            weights = weights / weights.sum(dim=-1, keepdim=True)

            for b in range(B):
                neighbor_vals = subgraph_pred[b][nn_idx]
                interp = (neighbor_vals * weights.unsqueeze(-1)).sum(dim=1)
                full_pred[b, missing_idx[start:end]] = interp

        return full_pred

    def scatter_back_to_subgraph(
        self,
        full_state: torch.Tensor,
    ) -> torch.Tensor:
        """For delayed writeback: gather updated state back onto the subgraph
        from the full grid after a writeback.
        """
        return self.gather_state(full_state)
