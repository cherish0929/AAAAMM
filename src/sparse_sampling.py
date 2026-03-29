"""
Learnable Sparse Sampling module for PhysGTO (v3 — memory-optimized).

Key memory optimizations vs v2:
  - SamplingNetwork: chunked forward to avoid holding (B, N, hidden) all at once
  - soft_topk on chunked logits: never materialize full (B, N) softmax
  - Scorer uses gradient checkpointing to trade compute for memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ═══════════════════════════════════════════════════════════════════════
# KNN graph builder with distance cutoff
# ═══════════════════════════════════════════════════════════════════════

def build_knn_graph(pos: torch.Tensor,
                    k: int = 16,
                    radius_cutoff: float = 0.0) -> torch.Tensor:
    """
    Build KNN graph with optional distance cutoff.

    Args:
        pos:            (B, K, 3) node positions
        k:              neighbors per node (clamped to K-1)
        radius_cutoff:  prune edges beyond this distance (0 = disabled)
    Returns:
        edges: (B, K*k, 2) int64
    """
    B, K_pts, D = pos.shape
    k = min(k, K_pts - 1)

    # Pairwise squared distances: (B, K, K)  — K is sparse, so this is fine
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist_sq = (diff ** 2).sum(-1)

    diag_mask = torch.eye(K_pts, device=pos.device, dtype=torch.bool).unsqueeze(0)
    dist_sq = dist_sq.masked_fill(diag_mask, float('inf'))

    knn_dist_sq, knn_idx = dist_sq.topk(k, dim=-1, largest=False)
    src = torch.arange(K_pts, device=pos.device).view(1, -1, 1).expand(B, K_pts, k)

    if radius_cutoff > 0:
        valid = knn_dist_sq < (radius_cutoff ** 2)
        src = torch.where(valid, src, torch.zeros_like(src))
        knn_idx = torch.where(valid, knn_idx, torch.zeros_like(knn_idx))

    return torch.stack([src, knn_idx], dim=-1).reshape(B, K_pts * k, 2)


# ═══════════════════════════════════════════════════════════════════════
# Sampling Network — with chunked evaluation
# ═══════════════════════════════════════════════════════════════════════

class SamplingNetwork(nn.Module):
    """
    Per-node importance scorer.

    Memory optimization: when N is large, evaluates the MLP in chunks
    so that only (B, chunk, hidden) is held in memory at any time,
    rather than (B, N, hidden).
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(feat_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, chunk_size: int = 0) -> torch.Tensor:
        """
        x: (B, N, feat_dim) -> logits: (B, N)

        chunk_size: if > 0, process N in chunks of this size.
                    Set to 0 to process all at once (small N).
        """
        if chunk_size <= 0 or x.shape[1] <= chunk_size:
            return self.net(x).squeeze(-1)

        # Chunked evaluation — only one chunk of activations in memory at a time
        B, N, _ = x.shape
        logits_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = x[:, start:end]  # (B, chunk, feat_dim)
            if chunk.requires_grad:
                out = checkpoint(self.net, chunk, use_reentrant=False)
            else:
                out = self.net(chunk)
            logits_chunks.append(out.squeeze(-1))  # (B, chunk)
        return torch.cat(logits_chunks, dim=1)  # (B, N)


# ═══════════════════════════════════════════════════════════════════════
# Differentiable soft top-K
# ═══════════════════════════════════════════════════════════════════════

def soft_topk(logits: torch.Tensor, K: int,
              tau: float = 1.0, hard: bool = False):
    """
    Returns:
        weights_K: (B, K) soft weights with gradient path
        indices:   (B, K) selected indices
    """
    B, N = logits.shape
    K = min(K, N)

    if hard or not logits.requires_grad:
        _, indices = logits.topk(K, dim=-1)
        weights_K = torch.ones(B, K, device=logits.device, dtype=logits.dtype)
        return weights_K, indices

    # Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    perturbed = (logits + gumbel) / tau

    # Top-K on perturbed logits (no softmax on full N — saves memory)
    topk_vals, indices = perturbed.topk(K, dim=-1)  # (B, K)

    # Softmax only on the K selected values (not full N)
    weights_K = F.softmax(topk_vals, dim=-1) * K  # (B, K), sum ≈ K

    return weights_K, indices


# ═══════════════════════════════════════════════════════════════════════
# Gather utilities
# ═══════════════════════════════════════════════════════════════════════

def soft_gather(features, indices, weights_K):
    """Gather + scale by soft weights (gradient bridge)."""
    B, K = indices.shape
    D = features.shape[-1]
    idx = indices.unsqueeze(-1).expand(B, K, D)
    gathered = torch.gather(features, 1, idx)

    gathered = gathered * weights_K.unsqueeze(-1) # 缩放
    context = torch.einsum("bk, bkd->bd", weights_K, gathered) # B,D 内部融合信息
    return gathered + context.unsqueeze(1)


def gather_by_indices(features, indices):
    """Hard gather (no gradient through indices)."""
    B, K = indices.shape
    D = features.shape[-1]
    idx = indices.unsqueeze(-1).expand(B, K, D)
    return torch.gather(features, 1, idx)


# ═══════════════════════════════════════════════════════════════════════
# LearnableSampler
# ═══════════════════════════════════════════════════════════════════════

class LearnableSampler(nn.Module):
    """
    Full differentiable sparse sampling pipeline.

    Memory-aware: scorer_chunk_size controls per-chunk evaluation on full grid.
    """

    def __init__(self,
                 feat_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 knn_k: int = 16,
                 radius_cutoff: float = 0.0,
                 tau: float = 1.0,
                 scorer_chunk_size: int = 100000):
        super().__init__()
        self.scorer = SamplingNetwork(feat_dim, hidden_dim, n_layers)
        self.knn_k = knn_k
        self.radius_cutoff = radius_cutoff
        self.tau = tau
        self.scorer_chunk_size = scorer_chunk_size
        self.global_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, scorer_input, node_features, node_pos, K, hard=False):
        """
        Args:
            scorer_input:  (B, N, scorer_feat_dim)
            node_features: (B, N, D) features to gather
            node_pos:      (B, N, 3)
            K:             number of points to select
            hard:          deterministic mode (eval)
        Returns:
            sampled_features, sampled_pos, edges, weights_K, indices
        """
        logits = self.scorer(scorer_input, chunk_size=self.scorer_chunk_size)
        weights_K, indices = soft_topk(logits, K, self.tau, hard)

        sampled_features = soft_gather(node_features, indices, weights_K)
        global_feat = torch.einsum("bn, bnd->bd", torch.softmax(logits, dim=-1), node_features)

        alpha = torch.sigmoid(self.global_gate)
        sampled_features = sampled_features + alpha * global_feat.unsqueeze(1) # 全局梯度路径(门控机制)
        sampled_pos = gather_by_indices(node_pos, indices)
        edges = build_knn_graph(sampled_pos, self.knn_k, self.radius_cutoff)

        return sampled_features, sampled_pos, edges, weights_K, indices
