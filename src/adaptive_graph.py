"""
adaptive_graph.py
=================
Dynamic adaptive graph construction for LPBF physics field prediction.

Architecture overview:
  1. Backbone graph  – coarse, fixed, always present (Section: build_backbone_*)
  2. Active zone scoring – gradient + temporal change (Section: compute_active_score)
  3. Zone classification – core / ring / background (Section: classify_zones)
  4. Refinement point sampling – from high-res candidates (Section: sample_refinement_points)
  5. Hybrid edge construction – backbone-backbone fixed + local radius/KNN (Section: build_combined_edges)
  6. Graph refresh logic – called every K steps (Section: refresh_refinement_graph)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  1. Backbone graph construction (固定粗网格骨架图)
# ---------------------------------------------------------------------------

def build_backbone_indices(
    grid_shape: Tuple[int, int, int],
    backbone_stride: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Compute 1-D flat indices for the backbone (coarse) grid.

    The backbone is a regularly-strided subset of the full-resolution grid.
    Boundary points are always included to avoid losing edge information.

    Returns:
        backbone_indices: 1-D array of flat indices into the full grid.
        backbone_shape: (nx_b, ny_b, nz_b) of the backbone sub-grid.
    """
    gx, gy, gz = grid_shape
    sx, sy, sz = backbone_stride

    xs = list(range(0, gx, sx))
    ys = list(range(0, gy, sy))
    zs = list(range(0, gz, sz))
    # always include last point per axis
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
    """6-neighbor grid edges for the backbone sub-grid.

    This is identical to the existing `_build_grid_edges`, kept here for
    clarity and to decouple the backbone from the dataset module.

    Returns:
        edges: [E_bb, 2] long tensor (undirected, deduplicated).
    """
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


# ---------------------------------------------------------------------------
#  2. Active zone scoring (活跃区域评分)
# ---------------------------------------------------------------------------

def compute_active_score(
    pred_field: torch.Tensor,
    gt_field: torch.Tensor,
    prev_field: torch.Tensor,
    node_pos: torch.Tensor,
    gradient_weight: float = 0.6,
    temporal_weight: float = 0.4,
    gt_blend: float = 1.0,
) -> torch.Tensor:
    """Per-node activity score in [0, 1] based on spatial gradient + temporal change.

    The score blends contributions from the predicted field and the ground-truth
    field.  As training progresses, `gt_blend` decreases from 1 → 0 so the model
    increasingly relies on its own predictions for refinement decisions.

    Args:
        pred_field:  [N, C] predicted physical state (normalized).
        gt_field:    [N, C] ground-truth physical state (normalized), or None.
        prev_field:  [N, C] state at the previous time-step (normalized).
        node_pos:    [N, 3] spatial positions.
        gradient_weight: weight for the spatial gradient term.
        temporal_weight: weight for the temporal change term.
        gt_blend:    blend ratio for ground-truth (1 = use GT fully, 0 = pred only).

    Returns:
        score: [N] tensor in [0, 1].
    """
    # blend predicted and ground-truth fields for scoring
    if gt_field is not None and gt_blend > 0.0:
        field = gt_blend * gt_field + (1.0 - gt_blend) * pred_field
    else:
        field = pred_field

    N = field.shape[0]
    device = field.device

    # --- spatial gradient approximation via finite differences to neighbors ---
    # Use pairwise distances to 8 nearest neighbors (cheap approx with cdist chunk)
    spatial_grad = _approximate_spatial_gradient(field, node_pos)

    # --- temporal change rate ---
    temporal_change = (field - prev_field).abs().mean(dim=-1)  # [N]

    # normalize each term to [0, 1]
    spatial_grad = _safe_minmax(spatial_grad)
    temporal_change = _safe_minmax(temporal_change)

    score = gradient_weight * spatial_grad + temporal_weight * temporal_change
    return score.clamp(0.0, 1.0)


def _approximate_spatial_gradient(
    field: torch.Tensor,
    pos: torch.Tensor,
    k: int = 6,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Approximate per-node spatial gradient magnitude using KNN differences.

    Computes gradient as the mean absolute field difference to the k nearest
    neighbors, weighted by inverse distance.  Uses chunked cdist to keep
    memory bounded.

    Returns:
        grad_mag: [N] tensor.
    """
    N = pos.shape[0]
    device = pos.device
    k_use = min(k + 1, N)  # +1 because the nearest is the node itself

    grad_mag = torch.zeros(N, device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # [chunk, N]
        dists = torch.cdist(pos[start:end].unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
        # top-k nearest (excluding self)
        _, idx = dists.topk(k_use, dim=-1, largest=False)
        # drop self (index 0 in sorted)
        nn_idx = idx[:, 1:]  # [chunk, k]

        # field differences
        chunk_field = field[start:end]  # [chunk, C]
        neighbor_field = field[nn_idx]  # [chunk, k, C]
        diff = (neighbor_field - chunk_field.unsqueeze(1)).abs().mean(dim=-1)  # [chunk, k]

        # inverse-distance weighting
        nn_dists = torch.gather(dists, 1, nn_idx)  # [chunk, k]
        weights = 1.0 / (nn_dists + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        grad_mag[start:end] = (diff * weights).sum(dim=-1)

    return grad_mag


def _safe_minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize to [0, 1]; returns zeros if range is negligible."""
    xmin, xmax = x.min(), x.max()
    rng = xmax - xmin
    if rng < 1e-12:
        return torch.zeros_like(x)
    return (x - xmin) / rng


# ---------------------------------------------------------------------------
#  3. Zone classification (三层区域分类: core / ring / background)
# ---------------------------------------------------------------------------

def classify_zones(
    score: torch.Tensor,
    core_threshold: float = 0.6,
    ring_threshold: float = 0.2,
) -> torch.Tensor:
    """Classify nodes into 3 zones based on activity score.

    Returns:
        zone: [N] int tensor with values:
            2 = core  (high activity, dense refinement)
            1 = ring  (transition, medium refinement)
            0 = background (backbone only)
    """
    zone = torch.zeros_like(score, dtype=torch.long)
    zone[score >= ring_threshold] = 1  # ring
    zone[score >= core_threshold] = 2  # core
    return zone


# ---------------------------------------------------------------------------
#  4. Refinement point sampling (从全分辨率候选中采样加密点)
# ---------------------------------------------------------------------------

def sample_refinement_points(
    zone_full: torch.Tensor,
    backbone_mask: torch.Tensor,
    core_keep_ratio: float = 1.0,
    ring_keep_ratio: float = 0.5,
) -> torch.Tensor:
    """Select refinement point indices from the full-resolution candidate set.

    Backbone points are never duplicated here — they are always present in the
    combined graph.  This function only selects *additional* high-res points.

    Args:
        zone_full:     [N_full] zone labels (0/1/2) for every full-res node.
        backbone_mask: [N_full] bool, True for nodes already in the backbone.
        core_keep_ratio: fraction of non-backbone core nodes to keep.
        ring_keep_ratio: fraction of non-backbone ring nodes to keep.

    Returns:
        refinement_indices: 1-D long tensor of indices into the full-res grid
                            for the selected refinement points (excludes backbone).
    """
    non_backbone = ~backbone_mask
    selected = []

    # core points (zone == 2) that are NOT backbone
    core_candidates = torch.where(non_backbone & (zone_full == 2))[0]
    if core_candidates.numel() > 0:
        n_keep = max(1, int(core_candidates.numel() * core_keep_ratio))
        perm = torch.randperm(core_candidates.numel(), device=core_candidates.device)[:n_keep]
        selected.append(core_candidates[perm])

    # ring points (zone == 1) that are NOT backbone
    ring_candidates = torch.where(non_backbone & (zone_full == 1))[0]
    if ring_candidates.numel() > 0:
        n_keep = max(1, int(ring_candidates.numel() * ring_keep_ratio))
        perm = torch.randperm(ring_candidates.numel(), device=ring_candidates.device)[:n_keep]
        selected.append(ring_candidates[perm])

    if selected:
        return torch.cat(selected)
    return torch.zeros(0, dtype=torch.long, device=zone_full.device)


# ---------------------------------------------------------------------------
#  5. Hybrid edge construction
#     (backbone-backbone 固定模板 + refinement 局部半径图/KNN)
# ---------------------------------------------------------------------------

def build_combined_edges(
    backbone_pos: torch.Tensor,
    refinement_pos: torch.Tensor,
    backbone_edges_local: torch.Tensor,
    n_backbone: int,
    cfg_edges: dict,
) -> torch.Tensor:
    """Build the union edge set for the combined (backbone + refinement) graph.

    Node ordering convention in the combined graph:
        [0 .. n_backbone-1]  →  backbone nodes
        [n_backbone .. n_backbone + n_ref - 1]  →  refinement nodes

    Edge types constructed:
        (a) backbone ↔ backbone: pre-computed grid edges (passed in).
        (b) refinement ↔ refinement: local radius graph, optionally capped by KNN.
        (c) refinement → backbone: each refinement node connects to its
            nearest `refinement_backbone_k` backbone nodes.

    Args:
        backbone_pos:        [N_bb, 3] positions.
        refinement_pos:      [N_ref, 3] positions (may be empty).
        backbone_edges_local:[E_bb, 2] edges in local backbone indexing.
        n_backbone:          number of backbone nodes.
        cfg_edges:           dict with keys:
            refinement_local_radius, refinement_local_k,
            refinement_backbone_k, max_refinement_edges.

    Returns:
        combined_edges: [E_total, 2] long tensor in the combined node indexing.
    """
    device = backbone_pos.device
    parts = []

    # (a) backbone ↔ backbone (already in local backbone indices, no shift needed)
    if backbone_edges_local.numel() > 0:
        parts.append(backbone_edges_local.to(device))

    n_ref = refinement_pos.shape[0] if refinement_pos is not None else 0
    if n_ref == 0:
        if parts:
            return torch.cat(parts, dim=0)
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    ref_offset = n_backbone  # refinement nodes start after backbone

    # (b) refinement ↔ refinement: local radius graph with KNN cap
    radius = cfg_edges.get("refinement_local_radius", 0.08)
    local_k = cfg_edges.get("refinement_local_k", 8)
    max_ref_edges = cfg_edges.get("max_refinement_edges", 50000)

    ref_edges = _build_local_radius_knn(
        refinement_pos, radius, local_k, max_ref_edges
    )
    if ref_edges.numel() > 0:
        ref_edges = ref_edges + ref_offset  # shift to combined indexing
        parts.append(ref_edges)

    # (c) refinement → backbone: nearest backbone neighbors
    rb_k = cfg_edges.get("refinement_backbone_k", 3)
    rb_edges = _build_cross_knn(
        query_pos=refinement_pos,
        target_pos=backbone_pos,
        k=rb_k,
        query_offset=ref_offset,
        target_offset=0,
    )
    if rb_edges.numel() > 0:
        parts.append(rb_edges)

    if parts:
        return torch.cat(parts, dim=0)
    return torch.zeros((0, 2), dtype=torch.long, device=device)


def _build_local_radius_knn(
    pos: torch.Tensor,
    radius: float,
    k: int,
    max_edges: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Radius graph with KNN cap among refinement points.

    For each node, connects to neighbors within `radius` but keeps at most
    `k` nearest.  Processes in chunks to avoid OOM on large point sets.

    Returns:
        edges: [E, 2] long tensor in local indexing (0-based within pos).
    """
    N = pos.shape[0]
    if N <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=pos.device)

    k_use = min(k + 1, N)
    edge_list = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = torch.cdist(pos[start:end].unsqueeze(0), pos.unsqueeze(0)).squeeze(0)  # [chunk, N]

        # mask beyond radius
        within = dists <= radius
        # self-loop mask
        self_mask = torch.zeros_like(within)
        for i in range(end - start):
            self_mask[i, start + i] = True
        within = within & ~self_mask

        # for each row, keep at most k
        for i in range(end - start):
            cands = torch.where(within[i])[0]
            if cands.numel() == 0:
                continue
            if cands.numel() > k:
                d = dists[i, cands]
                _, topk_idx = d.topk(k, largest=False)
                cands = cands[topk_idx]
            src = torch.full_like(cands, start + i)
            edge_list.append(torch.stack([src, cands], dim=1))

    if not edge_list:
        return torch.zeros((0, 2), dtype=torch.long, device=pos.device)

    edges = torch.cat(edge_list, dim=0)

    # cap total refinement edges
    if edges.shape[0] > max_edges:
        perm = torch.randperm(edges.shape[0], device=edges.device)[:max_edges]
        edges = edges[perm]

    return edges


def _build_cross_knn(
    query_pos: torch.Tensor,
    target_pos: torch.Tensor,
    k: int,
    query_offset: int,
    target_offset: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Cross-set KNN: each query point connects to its k nearest target points.

    Produces bidirectional edges (both directions).

    Returns:
        edges: [E, 2] long tensor in combined indexing.
    """
    N_q = query_pos.shape[0]
    N_t = target_pos.shape[0]
    if N_q == 0 or N_t == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=query_pos.device)

    k_use = min(k, N_t)
    edge_list = []

    for start in range(0, N_q, chunk_size):
        end = min(start + chunk_size, N_q)
        dists = torch.cdist(
            query_pos[start:end].unsqueeze(0),
            target_pos.unsqueeze(0),
        ).squeeze(0)  # [chunk, N_t]

        _, nn_idx = dists.topk(k_use, dim=-1, largest=False)  # [chunk, k_use]

        chunk_len = end - start
        src = torch.arange(start, end, device=query_pos.device).unsqueeze(1).expand(-1, k_use)
        # forward: query → target
        fwd = torch.stack([src.reshape(-1) + query_offset, nn_idx.reshape(-1) + target_offset], dim=1)
        # backward: target → query
        bwd = torch.stack([nn_idx.reshape(-1) + target_offset, src.reshape(-1) + query_offset], dim=1)
        edge_list.append(fwd)
        edge_list.append(bwd)

    if not edge_list:
        return torch.zeros((0, 2), dtype=torch.long, device=query_pos.device)
    return torch.cat(edge_list, dim=0)


# ---------------------------------------------------------------------------
#  6. Graph refresh orchestrator (每 K 步刷新一次 refinement graph)
# ---------------------------------------------------------------------------

class AdaptiveGraphManager:
    """Manages backbone + periodic refinement graph refresh during rollout.

    Usage during autoregressive inference:

        mgr = AdaptiveGraphManager(cfg, backbone_data, fullres_data, device)

        for t in range(T):
            if mgr.should_refresh(t):
                mgr.refresh(pred_field, gt_field, prev_field, epoch)

            combined_pos, combined_edges, combined_state = mgr.get_graph(pred_state)
            ...  # run GNN on combined graph
            full_pred = mgr.writeback(combined_pred)
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

        # backbone meta
        self.backbone_indices_np = backbone_indices
        self.backbone_shape = backbone_shape
        self.n_backbone = len(backbone_indices)
        self.backbone_edges = backbone_edges.to(device)

        # full resolution meta
        self.fullres_pos = fullres_pos.to(device)         # [N_full, 3]
        self.n_full = fullres_pos.shape[0]
        self.fullres_grid_shape = fullres_grid_shape

        # boolean mask: True for backbone nodes in the full grid
        self.backbone_mask = torch.zeros(self.n_full, dtype=torch.bool, device=device)
        bb_idx_tensor = torch.from_numpy(backbone_indices).long().to(device)
        self.backbone_mask[bb_idx_tensor] = True
        self.bb_idx_tensor = bb_idx_tensor

        # backbone positions (in full-grid coordinates)
        self.backbone_pos = fullres_pos[bb_idx_tensor]    # [N_bb, 3]

        # mapping from full-grid index → backbone local index
        self.full_to_bb = torch.full((self.n_full,), -1, dtype=torch.long, device=device)
        self.full_to_bb[bb_idx_tensor] = torch.arange(self.n_backbone, device=device)

        # refinement state (updated on refresh)
        self.refinement_full_indices: Optional[torch.Tensor] = None  # indices in full grid
        self.refinement_pos: Optional[torch.Tensor] = None
        self.n_refinement = 0
        self.combined_edges: Optional[torch.Tensor] = None

        # combined-graph → full-grid index mapping
        self.combined_to_full: Optional[torch.Tensor] = None

        # config shortcuts
        self.refresh_K = cfg.get("refinement", {}).get("refresh_every_K", 4)
        self.az_cfg = cfg.get("active_zone", {})
        self.edge_cfg = cfg.get("edges", {})
        self.wb_cfg = cfg.get("writeback", {})

    # ---- refresh logic ----------------------------------------------------

    def should_refresh(self, t: int) -> bool:
        """Returns True on steps 0, K, 2K, ..."""
        return t % self.refresh_K == 0

    def get_gt_blend(self, epoch: int) -> float:
        """Linear blend from gt_blend_start → gt_blend_end over warmup epochs."""
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
        """Rebuild the refinement graph based on current activity.

        All tensors are [N_full, C] on device, in *normalized* space.
        """
        gt_blend = self.get_gt_blend(epoch)

        # 2. compute per-node activity score
        score = compute_active_score(
            pred_field=pred_field,
            gt_field=gt_field,
            prev_field=prev_field,
            node_pos=self.fullres_pos,
            gradient_weight=self.az_cfg.get("gradient_weight", 0.6),
            temporal_weight=self.az_cfg.get("temporal_weight", 0.4),
            gt_blend=gt_blend,
        )

        # 3. zone classification (core=2, ring=1, bg=0)
        zone = classify_zones(
            score,
            core_threshold=self.az_cfg.get("core_threshold", 0.6),
            ring_threshold=self.az_cfg.get("ring_threshold", 0.2),
        )

        # 4. sample refinement points (non-backbone)
        ref_indices = sample_refinement_points(
            zone,
            self.backbone_mask,
            core_keep_ratio=self.az_cfg.get("core_keep_ratio", 1.0),
            ring_keep_ratio=self.az_cfg.get("ring_keep_ratio", 0.5),
        )
        self.refinement_full_indices = ref_indices
        self.n_refinement = ref_indices.numel()

        if self.n_refinement > 0:
            self.refinement_pos = self.fullres_pos[ref_indices]
        else:
            self.refinement_pos = torch.zeros((0, 3), device=self.device)

        # 5. build combined edges
        self.combined_edges = build_combined_edges(
            backbone_pos=self.backbone_pos,
            refinement_pos=self.refinement_pos,
            backbone_edges_local=self.backbone_edges,
            n_backbone=self.n_backbone,
            cfg_edges=self.edge_cfg,
        )

        # 6. combined → full index mapping
        #    [bb_0, bb_1, ..., bb_{N_bb-1}, ref_0, ref_1, ..., ref_{N_ref-1}]
        combined_full = torch.cat([
            self.bb_idx_tensor,
            ref_indices if ref_indices.numel() > 0 else torch.zeros(0, dtype=torch.long, device=self.device),
        ])
        self.combined_to_full = combined_full

    # ---- graph assembly ---------------------------------------------------

    def get_combined_count(self) -> int:
        return self.n_backbone + self.n_refinement

    def gather_state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract state for the combined graph nodes from the full-grid state.

        Args:
            full_state: [B, N_full, C] or [N_full, C].

        Returns:
            combined_state: [B, N_combined, C] or [N_combined, C].
        """
        if self.combined_to_full is None:
            # fallback: backbone only
            idx = self.bb_idx_tensor
        else:
            idx = self.combined_to_full

        if full_state.dim() == 3:
            # batched
            return full_state[:, idx]
        return full_state[idx]

    def get_combined_pos(self) -> torch.Tensor:
        """Return [N_combined, 3] positions for the current combined graph."""
        if self.n_refinement > 0 and self.refinement_pos is not None:
            return torch.cat([self.backbone_pos, self.refinement_pos], dim=0)
        return self.backbone_pos

    def get_combined_edges(self) -> torch.Tensor:
        """Return [E, 2] edges for the current combined graph."""
        if self.combined_edges is not None:
            return self.combined_edges
        return self.backbone_edges

    # ---- writeback: combined prediction → full grid -----------------------

    def writeback(
        self,
        combined_pred: torch.Tensor,
        full_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Write predictions from the combined graph back to the full grid.

        For nodes present in the combined graph, their predicted values are
        directly placed.  For nodes absent (background, not in backbone),
        inverse-distance weighted (IDW) interpolation from the combined graph
        is used.

        Args:
            combined_pred: [B, N_combined, C] predicted state on combined graph.
            full_prev:     [B, N_full, C] previous full-grid state (used as
                           fallback / initialization).

        Returns:
            full_pred: [B, N_full, C].
        """
        method = self.wb_cfg.get("method", "idw")
        B, _, C = combined_pred.shape
        full_pred = full_prev.clone()

        # direct copy for nodes in the combined graph
        idx = self.combined_to_full if self.combined_to_full is not None else self.bb_idx_tensor
        full_pred[:, idx] = combined_pred

        # interpolate missing nodes
        all_indices = torch.arange(self.n_full, device=self.device)
        present_mask = torch.zeros(self.n_full, dtype=torch.bool, device=self.device)
        present_mask[idx] = True
        missing_indices = all_indices[~present_mask]

        if missing_indices.numel() == 0:
            return full_pred

        if method == "idw":
            full_pred = self._idw_interpolate(
                full_pred, combined_pred, idx, missing_indices, B, C,
            )
        # else: leave as full_prev (nearest-copy fallback)

        return full_pred

    def _idw_interpolate(
        self,
        full_pred: torch.Tensor,
        combined_pred: torch.Tensor,
        present_idx: torch.Tensor,
        missing_idx: torch.Tensor,
        B: int,
        C: int,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Inverse-distance weighted interpolation for missing nodes.

        For each missing node, finds its k nearest *present* nodes and computes
        a weighted average of their predicted values.
        """
        power = self.wb_cfg.get("idw_power", 2.0)
        k = self.wb_cfg.get("idw_k_neighbors", 4)

        combined_pos = self.get_combined_pos()  # [N_combined, 3]
        missing_pos = self.fullres_pos[missing_idx]  # [N_missing, 3]
        N_miss = missing_pos.shape[0]
        k_use = min(k, combined_pos.shape[0])

        for start in range(0, N_miss, chunk_size):
            end = min(start + chunk_size, N_miss)
            dists = torch.cdist(
                missing_pos[start:end].unsqueeze(0),
                combined_pos.unsqueeze(0),
            ).squeeze(0)  # [chunk, N_combined]

            _, nn_idx = dists.topk(k_use, dim=-1, largest=False)  # [chunk, k]
            nn_dists = torch.gather(dists, 1, nn_idx)             # [chunk, k]

            weights = 1.0 / (nn_dists.pow(power) + 1e-10)        # [chunk, k]
            weights = weights / weights.sum(dim=-1, keepdim=True) # normalized

            # gather neighbor predictions and weighted-average
            for b in range(B):
                neighbor_vals = combined_pred[b][nn_idx]          # [chunk, k, C]
                interp = (neighbor_vals * weights.unsqueeze(-1)).sum(dim=1)  # [chunk, C]
                full_pred[b, missing_idx[start:end]] = interp

        return full_pred
