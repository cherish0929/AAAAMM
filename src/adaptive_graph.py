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
#  1. Active zone scoring  (基于物理量的连续打分: 温度 + 界面)
# ===================================================================

# Per-field normalization stats (mean, std) — same as dataset
_FIELD_STATS = {
    "T":              (5.2999e+02, 4.5454e+02),
    "Ux":             (4.0041e-05, 2.4173e-01),
    "Uy":             (-1.6900e-05, 2.5172e-01),
    "Uz":             (3.3602e-07, 1.1976e-01),
    "alpha.air":      (0, 1),
    "alpha.titanium": (0, 1),
    "gamma_liquid":   (0, 1),
}


def compute_active_score(
    pred_field: torch.Tensor,
    gt_field: Optional[torch.Tensor],
    prev_field: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    fields: Optional[List[str]] = None,
    field_stats: Optional[Dict] = None,
    T_ref: float = 600.0,
    T_high: float = 1500.0,
    interface_bonus: float = 0.5,
    air_discount: float = 0.5,
    gt_blend: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Per-node activity score in [0, 1] based on melt-pool physics.

    Scoring logic:
      1. Temperature score: smooth ramp from T_ref to T_high → [0, 1].
      2. Interface bonus: 4*alpha*(1-alpha), peaks at alpha=0.5.
         Multiplies score by (1 + interface_bonus) at the melt pool interface.
      3. Air discount: pure gas (alpha >= 0.9) gets its score multiplied by
         air_discount (< 1), reducing attention to pure gas regions.

    Combined: T_score * (1 + interface_bonus*4*a*(1-a)) * air_multiplier
    where air_multiplier = air_discount for alpha>=0.9, else 1.0.

    Result (at high T):
      - alpha ≈ 0.5 (interface) → highest score  (melt pool core)
      - alpha = 0   (solid)     → base score      (hot substrate)
      - alpha = 1   (gas)       → score * air_discount  (discounted)
    Result (at low T):
      - all alpha values         → ~0              (background)

    Fallback: if the field set does not contain both "T" and "alpha.air",
    uses temporal change |field_t - field_{t-1}| as a simple score.

    Args:
        T_ref:           raw temperature (K) below which score ≈ 0.
        T_high:          raw temperature (K) above which score saturates.
        interface_bonus: score multiplier bonus for the melt pool interface.
        air_discount:    score multiplier for pure gas regions (alpha >= 0.9).
                         Values < 1 reduce gas attention; 0 = ignore gas entirely.
    """
    if gt_field is not None and gt_blend > 0.0:
        field = gt_blend * gt_field + (1.0 - gt_blend) * pred_field
    else:
        field = pred_field

    fields = fields or []
    stats = field_stats or _FIELD_STATS

    has_T = "T" in fields
    has_alpha = "alpha.air" in fields

    if has_T and has_alpha:
        return _physics_score(field, fields, stats, T_ref, T_high, interface_bonus, air_discount)
    else:
        # fallback: temporal change (for non-standard field sets)
        temporal = (field - prev_field).abs().mean(dim=-1)
        return _safe_normalize(temporal)


def _physics_score(
    field: torch.Tensor,
    fields: List[str],
    stats: Dict,
    T_ref: float,
    T_high: float,
    interface_bonus: float,
    air_discount: float = 0.5,
) -> torch.Tensor:
    """Physics-based score using T and alpha.air channels.

    alpha.air = 0   → solid/liquid substrate (base score, no reduction)
    alpha.air = 1   → pure gas (score multiplied by air_discount < 1)
    alpha.air ∈ (0,1) → melt pool interface (highest score via interface_bonus)

    Score = T_score * (1 + interface_bonus * 4*a*(1-a)) * air_multiplier
    where air_multiplier = air_discount for pure gas (alpha >= 0.9), else 1.0

    At high T:
      alpha=0   (solid)     → T * 1.0  * 1.0           = T
      alpha=0.5 (interface) → T * (1 + interface_bonus) * 1.0
      alpha=1   (gas)       → T * 1.0  * air_discount   = T * air_discount
    """
    T_idx = fields.index("T")
    alpha_idx = fields.index("alpha.air")

    T_norm = field[:, T_idx]
    alpha_norm = field[:, alpha_idx]

    T_mean, T_std = stats.get("T", (5.2999e+02, 4.5454e+02))
    T_ref_norm = (T_ref - T_mean) / T_std if T_std != 0 else T_ref - T_mean
    T_high_norm = (T_high - T_mean) / T_std if T_std != 0 else T_high - T_mean

    # 1) Temperature score: smooth ramp [0, 1]
    T_score = _smooth_ramp(T_norm, T_ref_norm, T_high_norm)

    alpha_mean, alpha_std = stats.get("alpha.air", (0, 1))
    alpha_raw = alpha_norm * alpha_std + alpha_mean
    alpha_raw = alpha_raw.clamp(0.0, 1.0)

    # 2) Interface score: parabola peaking at alpha=0.5
    interface_score = 4.0 * alpha_raw * (1.0 - alpha_raw)

    # 3) Air discount: pure gas (alpha >= 0.9) gets multiplied by air_discount
    is_air = (alpha_raw >= 0.9).float()
    air_multiplier = 1.0 - (1.0 - air_discount) * is_air  # 1.0 normally, air_discount for gas

    # 4) Combined
    score = T_score * (1.0 + interface_bonus * interface_score) * air_multiplier

    # return score.clamp(0.0, 1.0)
    return score


def _smooth_ramp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth ramp from 0 to 1 over [lo, hi] using a sigmoid-like curve.

    Below lo → ~0, above hi → ~1, smooth transition in between.
    Uses a scaled sigmoid: sigmoid(6 * (x - mid) / (hi - lo)).
    """
    if hi - lo < 1e-12:
        return (x >= lo).float()
    mid = (lo + hi) / 2.0
    scale = 6.0 / (hi - lo)  # sigmoid reaches ~0.95 at hi, ~0.05 at lo
    return torch.sigmoid(scale * (x - mid))


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
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

        nb_local = full_to_local[nb_flat] # [n_zone, K]
        in_selected = (nb_local >= 0) & valid

        if compressed:
            # --- Compressed mode: per-zone neighbor budget ---
            if zone_val == 2:  # core
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

    # Vectorized base-6 detection: [K_full, 1, 3] == [1, 6, 3] → [K_full, 6]
    # .all(dim=-1) → [K_full, 6] bool; .any(dim=-1) → [K_full] bool
    is_base6 = (
        full_off_t.unsqueeze(1) == base6_off_t.unsqueeze(0)
    ).all(dim=-1).any(dim=-1)  # [K_full] bool

    # base6 mask: [n_zone, K_full]
    base6_mask = is_base6.unsqueeze(0).expand(n_zone, -1)
    extra_mask = ~base6_mask

    # base-6 edges (candidates)
    base_edges = in_selected & base6_mask

    # count base-6 per node
    base_count = base_edges.sum(dim=1)  # [n_zone]

    # If base-6 edges already exceed max_neighbors, trim via vectorized 2D topk
    base_over_budget = base_count > max_neighbors
    if base_over_budget.any():
        rand_base = torch.rand(n_zone, K_full, device=device)
        rand_base[~base_edges] = -1.0
        k_top_b = min(max_neighbors, K_full)
        top_vals_b, top_col_idx_b = rand_base.topk(k_top_b, dim=1)  # [n_zone, k_top_b]
        rank_pos_b = torch.arange(k_top_b, device=device).unsqueeze(0)  # [1, k_top_b]
        # keep entry if rank < max_neighbors and was a real candidate (score > -1)
        valid_b = (rank_pos_b < max_neighbors) & (top_vals_b > -1.0)
        new_base = torch.zeros(n_zone, K_full, dtype=torch.bool, device=device)
        over_mask_2d = base_over_budget.unsqueeze(1).expand(-1, k_top_b)
        new_base.scatter_(1, top_col_idx_b, valid_b & over_mask_2d)
        # restore rows that are NOT over-budget
        new_base[~base_over_budget] = base_edges[~base_over_budget]
        base_edges = new_base
        base_count = base_edges.sum(dim=1)

    # extra candidates
    extra_candidates = in_selected & extra_mask

    # budget remaining per node
    budget = (max_neighbors - base_count).clamp(min=0)  # [n_zone]

    # For each node, randomly sample from extras up to budget.
    # Vectorized: 2D topk then mask by per-row budget — no Python loop or .item().
    extra_count = extra_candidates.sum(dim=1)  # [n_zone]
    needs_trim = extra_count > budget

    if needs_trim.any():
        rand_scores = torch.rand(n_zone, K_full, device=device)
        rand_scores[~extra_candidates] = -1.0  # non-candidates rank last

        k_top = min(max_neighbors, K_full)
        top_vals, top_col_idx = rand_scores.topk(k_top, dim=1)  # [n_zone, k_top]
        rank_pos = torch.arange(k_top, device=device).unsqueeze(0)  # [1, k_top]
        # entry is valid if its rank < per-row budget AND it was a genuine candidate
        valid = (rank_pos < budget.unsqueeze(1)) & (top_vals > -1.0)

        new_extra = torch.zeros(n_zone, K_full, dtype=torch.bool, device=device)
        trim_mask_2d = needs_trim.unsqueeze(1).expand(-1, k_top)
        new_extra.scatter_(1, top_col_idx, valid & trim_mask_2d)
        # preserve rows that don't need trimming
        new_extra[~needs_trim] = extra_candidates[~needs_trim]
        extra_candidates = new_extra

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


def enforce_per_node_degree(
    edges: torch.Tensor,
    zone_labels_selected: torch.Tensor,
    core_max_degree: int = 8,
    ring_max_degree: int = 6,
    bg_max_degree: int = 6,
    num_nodes: int = 0,
    selected_indices: Optional[torch.Tensor] = None,
    grid_shape: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """Enforce strict per-node degree limits by zone.

    For each node whose total degree exceeds its zone's limit, drop edges
    with the following priority (highest kept, lowest dropped first):
      1. base-6 (face-adjacent) edges         — highest priority
      2. edges where the other endpoint has a higher zone value
      3. random tiebreaker

    Args:
        edges:                [E, 2] canonical edges (src < dst).
        zone_labels_selected: [N_sel] zone labels for the selected nodes.
        core_max_degree:      max edges per core node.
        ring_max_degree:      max edges per ring node.
        bg_max_degree:        max edges per background node.
        num_nodes:            number of selected nodes.
        selected_indices:     [N_sel] full-grid flat indices (needed for base6 detection).
        grid_shape:           (nx, ny, nz) of the full grid (needed for base6 detection).

    Returns:
        trimmed: [E', 2] edges respecting per-node degree limits.
    """
    if edges.numel() == 0 or num_nodes == 0:
        return edges

    device = edges.device
    N = num_nodes
    E = edges.shape[0]

    # Build per-node degree budget
    max_deg = torch.full((N,), bg_max_degree, dtype=torch.long, device=device)
    max_deg[zone_labels_selected == 1] = ring_max_degree
    max_deg[zone_labels_selected == 2] = core_max_degree

    # Compute current degree
    degree = torch.zeros(N, dtype=torch.long, device=device)
    degree.scatter_add_(0, edges[:, 0], torch.ones(E, dtype=torch.long, device=device))
    degree.scatter_add_(0, edges[:, 1], torch.ones(E, dtype=torch.long, device=device))

    over_budget = degree > max_deg
    if not over_budget.any():
        return edges

    # --- Edge priority scoring ---
    # Component 1: base-6 bonus (face-adjacent in grid = Manhattan distance 1)
    base6_bonus = torch.zeros(E, device=device)
    if selected_indices is not None and grid_shape is not None:
        nx, ny, nz = grid_shape
        src_full = selected_indices[edges[:, 0]]
        dst_full = selected_indices[edges[:, 1]]
        sx = src_full % nx;           dx = dst_full % nx
        sy = (src_full // nx) % ny;   dy = (dst_full // nx) % ny
        sz = src_full // (nx * ny);   dz = dst_full // (nx * ny)
        manhattan = (sx - dx).abs() + (sy - dy).abs() + (sz - dz).abs()
        base6_bonus[manhattan == 1] = 4.0  # strong bonus for face-adjacent

    # Component 2: other-endpoint zone value (higher zone neighbor = more useful)
    src_zone = zone_labels_selected[edges[:, 0]].float()
    dst_zone = zone_labels_selected[edges[:, 1]].float()
    neighbor_zone_bonus = (src_zone + dst_zone)  # 0~4

    # Component 3: random tiebreaker
    rand_jitter = torch.rand(E, device=device) * 0.9

    # Total priority: higher = keep
    edge_priority = base6_bonus + neighbor_zone_bonus + rand_jitter

    # Sort edges by priority ascending (drop lowest first)
    sorted_idx = edge_priority.argsort()

    # Move the greedy loop to CPU numpy to eliminate 2*E GPU→CPU syncs.
    # Semantics are identical: process edges in ascending priority order,
    # drop an edge if either endpoint is over-budget, decrement both degrees.
    edges_np      = edges.cpu().numpy()         # [E, 2]
    sorted_idx_np = sorted_idx.cpu().numpy()    # [E]
    max_deg_np    = max_deg.cpu().numpy()       # [N]
    cur_deg_np    = degree.cpu().numpy().copy() # [N]  mutable copy
    keep_np       = np.ones(E, dtype=np.bool_)

    for ei in sorted_idx_np:
        s = edges_np[ei, 0]
        d = edges_np[ei, 1]
        if cur_deg_np[s] > max_deg_np[s] or cur_deg_np[d] > max_deg_np[d]:
            keep_np[ei] = False
            cur_deg_np[s] -= 1
            cur_deg_np[d] -= 1

    keep_mask = torch.from_numpy(keep_np).to(device)
    return edges[keep_mask]


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
    backbone_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build cross-layer edges between fine (core/ring) nodes and their
    coarse backbone parent cells using regular-grid parent-child mapping.

    Only connects fine nodes to backbone parent nodes (not fine→fine).

    For each fine node at (x,y,z), its parent backbone node is at
    (x // sx, y // sy, z // sz) * stride.  We connect the fine node to
    the nearest backbone node(s) using the stride relationship.

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
    parent_x = ((fine_x + sx // 2) // sx * sx).clamp(0, nx - 1)
    parent_y = ((fine_y + sy // 2) // sy * sy).clamp(0, ny - 1)
    parent_z = ((fine_z + sz // 2) // sz * sz).clamp(0, nz - 1)

    parent_flat = parent_z * (nx * ny) + parent_y * nx + parent_x

    # check if parent is in our selected set
    parent_local = full_to_local[parent_flat]
    valid = (parent_local >= 0)

    # avoid self-loops
    valid = valid & (fine_local != parent_local)

    # ensure parent is actually a backbone node (not a fine node)
    if backbone_mask is not None:
        parent_is_bb = backbone_mask[parent_flat]
        valid = valid & parent_is_bb

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
        fields: Optional[List[str]] = None,
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
        self._fields = fields or []

        # IDW interpolation cache: invalidated on every refresh call.
        # Stores (refresh_count_key, nn_idx [N_miss, k], weights [N_miss, k]).
        self._idw_cache: Optional[tuple] = None
        self._refresh_count: int = 0

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

        # Step 1: compute activity score (physics-based: T + alpha.air)
        score = compute_active_score(
            pred_field=pred_field,
            gt_field=gt_field,
            prev_field=prev_field,
            grid_shape=self.fullres_grid_shape,
            fields=self._fields,
            T_ref=self.az_cfg.get("T_ref", 600.0),
            T_high=self.az_cfg.get("T_high", 1500.0),
            interface_bonus=self.az_cfg.get("interface_bonus", 0.5),
            air_discount=self.az_cfg.get("air_discount", 0.5),
            gt_blend=gt_blend,
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

        # Step 4: build intra-layer stencil edges
        if use_compressed:
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

        all_edges = intra_edges if intra_edges.numel() > 0 else \
            torch.zeros((0, 2), dtype=torch.long, device=self.device)

        # Step 5: enforce global edge budget on intra edges (compressed mode)
        if use_compressed and all_edges.numel() > 0:
            sel_zone = zone[selected]
            max_edges_ratio = self.compress_cfg.get("max_edges_ratio", 4.0)
            all_edges = enforce_edge_budget(
                edges=all_edges,
                zone_labels_selected=sel_zone,
                max_edges_ratio=max_edges_ratio,
                num_nodes=selected.shape[0],
            )

        # Step 6: enforce strict per-node degree limits on intra edges
        #         (base6-aware: face-adjacent edges are preferentially kept)
        if use_compressed and all_edges.numel() > 0:
            sel_zone = zone[selected]
            all_edges = enforce_per_node_degree(
                edges=all_edges,
                zone_labels_selected=sel_zone,
                core_max_degree=self.compress_cfg.get("core_max_neighbors", 12),
                ring_max_degree=self.compress_cfg.get("ring_max_neighbors", 8),
                bg_max_degree=max(1, int(6 * self.compress_cfg.get("bg_edge_keep_ratio", 0.7))),
                num_nodes=selected.shape[0],
                selected_indices=selected,
                grid_shape=self.fullres_grid_shape,
            )

        # Step 7: append cross-layer edges AFTER degree truncation
        #         (fine→backbone only, not fine→fine)
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
            backbone_mask=self.backbone_mask,
        )

        if cross_edges.numel() > 0:
            if use_compressed:
                cross_edges = canonicalize_and_dedup_edges(cross_edges)
            edge_parts = [all_edges, cross_edges] if all_edges.numel() > 0 else [cross_edges]
            all_edges = torch.cat(edge_parts, dim=0)
            if use_compressed:
                all_edges = canonicalize_and_dedup_edges(all_edges)

        # Step 7: remove isolated nodes (nodes with no edges)
        if all_edges.numel() > 0 and selected.shape[0] > 0:
            connected = torch.zeros(selected.shape[0], dtype=torch.bool, device=self.device)
            connected[all_edges[:, 0]] = True
            connected[all_edges[:, 1]] = True

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

        # Invalidate IDW cache: selected nodes have changed.
        self._refresh_count += 1
        self._idw_cache = None

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
        """IDW interpolation for missing nodes.

        nn_idx and weights are cached between writeback calls when the selected
        node set has not changed (i.e. between refreshes).  The cache is
        invalidated automatically at every refresh() call.
        """
        power = self.wb_cfg.get("idw_power", 2.0)
        k = self.wb_cfg.get("idw_k_neighbors", 4)

        selected_pos = self.get_selected_pos()
        missing_pos = self.fullres_pos[missing_idx]
        N_miss = missing_pos.shape[0]
        k_use = min(k, selected_pos.shape[0])

        # Cache validity: key matches current refresh count AND shape is consistent.
        cache_valid = (
            self._idw_cache is not None
            and self._idw_cache[0] == self._refresh_count
            and self._idw_cache[1].shape[0] == N_miss
        )

        if cache_valid:
            # Reuse pre-computed geometry — skip all cdist work.
            _, nn_idx_full, weights_full = self._idw_cache
            for b in range(B):
                neighbor_vals = subgraph_pred[b][nn_idx_full]           # [N_miss, k, C]
                interp = (neighbor_vals * weights_full.unsqueeze(-1)).sum(dim=1)
                full_pred[b, missing_idx] = interp
        else:
            # Full chunked cdist computation (same as original).
            # Accumulate nn_idx / weights for caching.
            all_nn_idx = torch.empty(N_miss, k_use, dtype=torch.long, device=self.device)
            all_weights = torch.empty(N_miss, k_use, device=self.device)

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

                all_nn_idx[start:end] = nn_idx
                all_weights[start:end] = weights

                for b in range(B):
                    neighbor_vals = subgraph_pred[b][nn_idx]
                    interp = (neighbor_vals * weights.unsqueeze(-1)).sum(dim=1)
                    full_pred[b, missing_idx[start:end]] = interp

            # Store cache for reuse until next refresh.
            self._idw_cache = (self._refresh_count, all_nn_idx, all_weights)

        return full_pred

    def scatter_back_to_subgraph(
        self,
        full_state: torch.Tensor,
    ) -> torch.Tensor:
        """For delayed writeback: gather updated state back onto the subgraph
        from the full grid after a writeback.
        """
        return self.gather_state(full_state)
