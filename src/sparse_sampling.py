"""
Learnable Sparse Sampling module for PhysGTO.

Provides:
  - SamplingNetwork: learns per-node importance scores from (state, position, physics fields)
  - soft_topk: differentiable soft top-K selection via Gumbel-Softmax reweighting
  - build_knn_graph: batched KNN graph construction using torch_cluster
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# KNN graph builder (uses torch_cluster for GPU-accelerated KNN)
# ---------------------------------------------------------------------------

def build_knn_graph(pos: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    Build a KNN graph for a batch of point clouds.

    Args:
        pos: (B, K, 3) sampled node positions
        k:   number of neighbors (clamped to K-1)

    Returns:
        edges: (B, K*k, 2) int64, each row = [src, dst]
    """
    B, K_pts, D = pos.shape
    k = min(k, K_pts - 1)

    # pairwise distances: (B, K, K)
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)      # (B, K, K, D)
    dist = (diff ** 2).sum(-1)                        # (B, K, K)

    # exclude self: set diagonal to inf
    diag_mask = torch.eye(K_pts, device=pos.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(diag_mask, float('inf'))

    # topk nearest neighbors
    _, idx = dist.topk(k, dim=-1, largest=False)      # (B, K, k)

    # build edge list
    src = torch.arange(K_pts, device=pos.device).view(1, -1, 1).expand(B, K_pts, k)
    edges = torch.stack([src, idx], dim=-1).reshape(B, K_pts * k, 2)  # (B, K*k, 2)
    return edges


# ---------------------------------------------------------------------------
# Sampling Network: outputs per-node importance score
# ---------------------------------------------------------------------------

class SamplingNetwork(nn.Module):
    """
    Learns importance scores from node features (state + position + optional physics fields).

    Input:  (B, N, feat_dim)   where feat_dim = state_dim + space_dim
    Output: (B, N)             importance logits (unnormalized)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(feat_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, feat_dim)
        Returns:
            logits: (B, N) importance logits
        """
        return self.net(x).squeeze(-1)  # (B, N)


# ---------------------------------------------------------------------------
# Differentiable soft top-K selection
# ---------------------------------------------------------------------------

def soft_topk(logits: torch.Tensor,
              K: int,
              tau: float = 1.0,
              hard: bool = False) -> torch.Tensor:
    """
    Differentiable soft top-K via Gumbel-Softmax weighted selection.

    During training:
      - Applies Gumbel noise + temperature-scaled softmax -> soft weights (B, N)
      - Returns the top-K weights (renormalized), which can be used to form
        weighted combinations of node features.
    During eval:
      - Deterministic top-K selection (hard=True).

    Args:
        logits: (B, N) raw importance scores
        K:      number of points to select
        tau:    temperature (lower = sharper)
        hard:   if True, returns one-hot-like weights (straight-through)

    Returns:
        weights: (B, N) with exactly K non-zero entries per batch
        indices: (B, K) selected indices (long)
    """
    B, N = logits.shape
    K = min(K, N)

    if hard or not logits.requires_grad:
        # Deterministic: just take top-K
        _, indices = logits.topk(K, dim=-1)           # (B, K)
        weights = torch.zeros_like(logits)              # (B, N)
        weights.scatter_(1, indices, 1.0)
        return weights, indices

    # Gumbel-Softmax: add Gumbel noise, then softmax
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    perturbed = (logits + gumbel_noise) / tau
    soft_weights = F.softmax(perturbed, dim=-1)        # (B, N)

    # Select top-K indices based on soft weights
    _, indices = soft_weights.topk(K, dim=-1)           # (B, K)

    # Create mask and apply straight-through if needed
    mask = torch.zeros_like(logits)
    mask.scatter_(1, indices, 1.0)                      # (B, N)

    # Straight-through: keep gradients flowing through soft_weights
    weights = soft_weights * mask
    # Renormalize so weights sum to K (each selected point ~ weight 1)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8) * K

    return weights, indices


def gather_by_indices(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather features at selected indices.

    Args:
        features: (B, N, D) full feature tensor
        indices:  (B, K)    selected indices

    Returns:
        selected: (B, K, D)
    """
    B, K = indices.shape
    D = features.shape[-1]
    idx_expanded = indices.unsqueeze(-1).expand(B, K, D)  # (B, K, D)
    return torch.gather(features, 1, idx_expanded)


# ---------------------------------------------------------------------------
# Convenience: full sampling pipeline
# ---------------------------------------------------------------------------

class LearnableSampler(nn.Module):
    """
    Full learnable sparse sampling pipeline.

    1. SamplingNetwork -> importance logits
    2. soft_topk -> select K points
    3. Gather features & positions at selected points
    4. Build KNN graph on selected points
    """

    def __init__(self,
                 feat_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 knn_k: int = 16,
                 tau: float = 1.0):
        super().__init__()
        self.scorer = SamplingNetwork(feat_dim, hidden_dim, n_layers)
        self.knn_k = knn_k
        self.tau = tau

    def forward(self,
                node_features: torch.Tensor,
                node_pos: torch.Tensor,
                K: int,
                hard: bool = False):
        """
        Args:
            node_features: (B, N, feat_dim)  - concatenation of state + pos + physics
            node_pos:      (B, N, 3)         - spatial positions
            K:             int               - number of points to select
            hard:          bool              - deterministic selection (eval mode)

        Returns:
            sampled_features: (B, K, feat_dim)
            sampled_pos:      (B, K, 3)
            edges:            (B, K*knn_k, 2)
            weights:          (B, N)  soft weights (for reconstruction loss)
            indices:          (B, K)  selected indices
        """
        # score all nodes
        logits = self.scorer(node_features)            # (B, N)

        # differentiable selection
        weights, indices = soft_topk(logits, K, tau=self.tau, hard=hard)

        # gather at selected indices
        sampled_features = gather_by_indices(node_features, indices)  # (B, K, feat_dim)
        sampled_pos = gather_by_indices(node_pos, indices)            # (B, K, 3)

        # build graph on sparse points
        edges = build_knn_graph(sampled_pos, k=self.knn_k)            # (B, K*knn_k, 2)

        return sampled_features, sampled_pos, edges, weights, indices
