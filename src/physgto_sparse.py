"""
PhysGTO-Sparse: Learnable Sparse Sampling + Operator Learning for LPBF.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Input: state_in (B, N, C), node_pos (B, N, 3), conditions, time  │
  └──────────────────────┬──────────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Sampler │  LearnableSampler: N -> K points
                    └────┬────┘
                         │  sampled_state (B, K, C), sampled_pos (B, K, 3)
                         │  knn_edges (B, K*knn_k, 2)
                    ┌────▼────┐
                    │ Encoder │  Node/Edge embedding on sparse graph
                    └────┬────┘
                         │  V (B, K, enc_dim), E (B, ne, enc_dim)
                    ┌────▼────┐
                    │  Mixer  │  N_block x (GNN + Attention + FFN)
                    └────┬────┘
                         │  V_all (B, N_block, K, enc_dim)
                    ┌────▼────┐
                    │ Decoder │  Two options:
                    │  (A)    │    MLP + Fourier: output on K points -> interpolate to N
                    │  (B)    │    Cross-Attention: query = full grid -> output on N
                    └────┬────┘
                         │  state_pred (B, N, C)
  ┌──────────────────────▼──────────────────────────────────────────────┐
  │  Output: state_pred (B, N, C)                                      │
  └─────────────────────────────────────────────────────────────────────┘

All tensor shapes are annotated inline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint

from .sparse_sampling import LearnableSampler, gather_by_indices, build_knn_graph


# ═══════════════════════════════════════════════════════════════════════
# Shared utilities (same as physgto_res.py)
# ═══════════════════════════════════════════════════════════════════════

def get_edge_info(edges, node_pos):
    """Compute edge features from node positions.
    edges: (B, ne, 2), node_pos: (B, N, 3)
    Returns E: (B, ne, 2*space+1)
    """
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True) + 1e-8)
    E = torch.cat([d, -d, norm], dim=-1)
    return E


def FourierEmbedding(pos, pos_start, pos_length):
    """Fourier positional encoding.
    pos: (..., D) -> (..., D * (2*pos_length) + D)
    """
    original_shape = pos.shape
    new_pos = pos.reshape(-1, original_shape[-1])
    index = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq = 2 ** index * torch.pi
    cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    embedding = torch.cat([cos_feat, sin_feat], dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, pos], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, layer_norm=True,
                 n_hidden=1, hidden_size=128, act='SiLU'):
        super().__init__()
        if act == 'GELU':
            self.act = nn.GELU()
        elif act == 'SiLU':
            self.act = nn.SiLU()
        elif act == 'PReLU':
            self.act = nn.PReLU()

        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), self.act]
            for _ in range(1, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))
        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


# ═══════════════════════════════════════════════════════════════════════
# Attention with Learnable Query Tokens (same as original Atten)
# ═══════════════════════════════════════════════════════════════════════

class Atten(nn.Module):
    def __init__(self, n_token=128, c_dim=128, n_heads=4):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(n_token, c_dim))
        self.attention1 = nn.MultiheadAttention(embed_dim=c_dim, num_heads=n_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=c_dim, num_heads=n_heads, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=c_dim, num_heads=n_heads, batch_first=True)

    def forward(self, W0):
        # W0: (B, K, c_dim)
        B = W0.shape[0]
        Q = self.Q.unsqueeze(0).expand(B, -1, -1)        # (B, n_token, c_dim)
        W, _ = self.attention1(Q, W0, W0)                 # (B, n_token, c_dim)
        W, _ = self.attention2(W, W, W)                    # (B, n_token, c_dim)
        W, _ = self.attention3(W0, W, W)                   # (B, K, c_dim)
        return W


# ═══════════════════════════════════════════════════════════════════════
# GNN on sparse graph
# ═══════════════════════════════════════════════════════════════════════

class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        output_size = output_size or node_size

        self.f_edge = MLP(
            input_size=edge_size + node_size * 2,
            n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU',
            output_size=edge_size
        )
        self.f_node = MLP(
            input_size=edge_size + node_size,
            n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU',
            output_size=output_size
        )

    def forward(self, V, E, edges):
        """
        V: (B, K, node_size), E: (B, ne, edge_size), edges: (B, ne, 2)
        """
        B, K, _ = V.shape
        senders = torch.gather(V, 1, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, 1, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        edge_embeddings_0, edge_embeddings_1 = edge_embeddings.chunk(2, dim=-1)
        feat0, feat1 = edge_embeddings_0.shape[-1], edge_embeddings_1.shape[-1]

        col_0 = edges[..., 0].unsqueeze(-1).expand(-1, -1, feat0)
        col_1 = edges[..., 1].unsqueeze(-1).expand(-1, -1, feat1)

        edge_mean_0 = scatter_mean(edge_embeddings_0, col_0, dim=1, dim_size=K)
        edge_mean_1 = scatter_mean(edge_embeddings_1, col_1, dim=1, dim_size=K)

        edge_mean = torch.cat([edge_mean_0, edge_mean_1], dim=-1)
        node_inpt = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


# ═══════════════════════════════════════════════════════════════════════
# MixerBlock: GNN + Attention + FFN on sparse K nodes
# ═══════════════════════════════════════════════════════════════════════

class MixerBlock(nn.Module):
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        node_size = enc_dim + enc_s_dim

        self.gnn = GNN(
            node_size=node_size,
            edge_size=enc_dim,
            output_size=enc_dim,
            layer_norm=True
        )
        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.mha = Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head)
        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, 2 * enc_dim),
            nn.SiLU(),
            nn.Linear(2 * enc_dim, enc_dim)
        )

    def forward(self, V, E, edges, s_enc):
        """
        V: (B, K, enc_dim), E: (B, ne, enc_dim), edges: (B, ne, 2)
        s_enc: (B, K, enc_s_dim) positional encoding of sampled nodes
        """
        V_in = torch.cat([V, s_enc], dim=-1)            # (B, K, enc_dim + enc_s_dim)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        V = V + v
        V = V + self.mha(self.ln1(V))
        V = V + self.ffn(self.ln2(V))
        return V, E


# ═══════════════════════════════════════════════════════════════════════
# Encoder: node + edge embedding on sparse graph
# ═══════════════════════════════════════════════════════════════════════

class SparseEncoder(nn.Module):
    def __init__(self, space_size=3, state_size=4, enc_dim=128,
                 enc_t_dim=11, enc_c_dim=12):
        super().__init__()
        self.fv1 = MLP(input_size=state_size + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        """
        node_pos:  (B, K, 3)      - sampled positions
        state_in:  (B, K, C)      - sampled state
        time_i:    (B, enc_t_dim) - time encoding
        conditions:(B, enc_c_dim) - condition encoding
        edges:     (B, ne, 2)     - KNN graph edges
        Returns V: (B, K, enc_dim), E: (B, ne, enc_dim)
        """
        x = torch.cat([state_in, node_pos], dim=-1)       # (B, K, C+3)
        time_enc = self.fv_time(time_i)                    # (B, enc_dim)
        cond_enc = self.fv_cond(conditions)                # (B, enc_dim)
        V = self.fv1(x) + time_enc.unsqueeze(-2) + cond_enc.unsqueeze(-2)  # (B, K, enc_dim)
        E = self.fe(get_edge_info(edges, node_pos))                         # (B, ne, enc_dim)
        return V, E


# ═══════════════════════════════════════════════════════════════════════
# Mixer: stack of MixerBlocks
# ═══════════════════════════════════════════════════════════════════════

class Mixer(nn.Module):
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock(enc_dim=enc_dim, n_head=n_head, n_token=n_token, enc_s_dim=enc_s_dim)
            for _ in range(N)
        ])

    def forward(self, V, E, edges, pos_enc):
        """
        V: (B, K, enc_dim), E: (B, ne, enc_dim), edges: (B, ne, 2)
        pos_enc: (B, K, enc_s_dim)
        Returns V_all: (B, N_block, K, enc_dim)
        """
        V_all = []
        for block in self.blocks:
            V, E = block(V, E, edges, pos_enc)
            V_all.append(V)
        return torch.stack(V_all, dim=1)  # (B, N_block, K, enc_dim)


# ═══════════════════════════════════════════════════════════════════════
# Decoder Option A: MLP + Fourier (output on K sparse points)
# ═══════════════════════════════════════════════════════════════════════

class SparseMLPDecoder(nn.Module):
    """
    Decodes latent on K sparse points -> delta on K points.
    Reconstruction to full N is done via interpolation (outside this module).

    Input:  V_all (B, N_block, K, enc_dim) + pos_enc (B, K, enc_s_dim)
    Output: delta (B, K, out_dim)
    """

    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10, state_size=1):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(N_block * enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc):
        """
        V_all:   (B, N_block, K, enc_dim)
        pos_enc: (B, K, enc_s_dim)
        Returns: (B, K, out_dim)
        """
        B, N_block, K, enc_dim = V_all.shape
        V_flat = V_all.permute(0, 2, 1, 3).reshape(B, K, -1)  # (B, K, N_block*enc_dim)
        return self.delta_net(torch.cat([V_flat, pos_enc], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# Decoder Option B: Cross-Attention Decoder (query = full grid)
# ═══════════════════════════════════════════════════════════════════════

class CrossAttentionDecoder(nn.Module):
    """
    Uses full-grid positions as queries to attend over sparse latent tokens.

    Input:
      V_all:        (B, N_block, K, enc_dim) - sparse latent from mixer
      pos_enc_K:    (B, K, enc_s_dim)        - Fourier encoding of sampled positions
      pos_enc_full: (B, N, enc_s_dim)        - Fourier encoding of ALL positions

    Output: delta (B, N, out_dim)  - predicted delta on full grid
    """

    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10,
                 state_size=1, n_heads=4):
        super().__init__()

        # Compress multi-block latent into single vector per node
        self.latent_proj = nn.Sequential(
            nn.Linear(N_block * enc_dim, enc_dim),
            nn.SiLU()
        )

        # Query projection: full grid position encoding -> query
        self.query_proj = nn.Linear(enc_s_dim, enc_dim)

        # Key/Value projection: sparse latent + pos -> key, value
        self.kv_proj = nn.Linear(enc_dim + enc_s_dim, enc_dim * 2)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=enc_dim, num_heads=n_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(enc_dim)

        # Output head
        self.out_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc_K, pos_enc_full):
        """
        V_all:        (B, N_block, K, enc_dim)
        pos_enc_K:    (B, K, enc_s_dim)
        pos_enc_full: (B, N, enc_s_dim)
        Returns:      (B, N, out_dim)
        """
        B, N_block, K, enc_dim = V_all.shape

        # Flatten multi-block latent
        V_flat = V_all.permute(0, 2, 1, 3).reshape(B, K, -1)  # (B, K, N_block*enc_dim)
        V_proj = self.latent_proj(V_flat)                       # (B, K, enc_dim)

        # Key/Value from sparse latent + position
        kv_input = torch.cat([V_proj, pos_enc_K], dim=-1)      # (B, K, enc_dim+enc_s_dim)
        kv = self.kv_proj(kv_input)                              # (B, K, 2*enc_dim)
        key, value = kv.chunk(2, dim=-1)                         # each (B, K, enc_dim)

        # Query from full grid positions
        query = self.query_proj(pos_enc_full)                    # (B, N, enc_dim)

        # Cross attention: full grid queries attend to sparse keys
        out, _ = self.cross_attn(query, key, value)              # (B, N, enc_dim)
        out = self.ln1(out + query)                              # residual + norm

        return self.out_head(out)                                # (B, N, out_dim)


# ═══════════════════════════════════════════════════════════════════════
# IDW interpolation: sparse K -> full N
# ═══════════════════════════════════════════════════════════════════════

def idw_interpolate(sparse_values: torch.Tensor,
                    sparse_pos: torch.Tensor,
                    full_pos: torch.Tensor,
                    power: float = 2.0,
                    k_neighbors: int = 8) -> torch.Tensor:
    """
    Inverse-Distance-Weighted interpolation from K sparse points to N full points.

    Args:
        sparse_values: (B, K, C) values at sparse points
        sparse_pos:    (B, K, 3) sparse positions
        full_pos:      (B, N, 3) full grid positions
        power:         IDW exponent
        k_neighbors:   number of nearest sparse points to use per full point

    Returns:
        full_values: (B, N, C)
    """
    B, N, D = full_pos.shape
    K = sparse_pos.shape[1]
    k_neighbors = min(k_neighbors, K)

    # Pairwise distances: (B, N, K)
    diff = full_pos.unsqueeze(2) - sparse_pos.unsqueeze(1)   # (B, N, K, 3)
    dist = (diff ** 2).sum(-1).sqrt().clamp(min=1e-8)         # (B, N, K)

    # Top-k nearest neighbors
    _, nn_idx = dist.topk(k_neighbors, dim=-1, largest=False)  # (B, N, k_neighbors)
    nn_dist = torch.gather(dist, 2, nn_idx)                    # (B, N, k_neighbors)

    # IDW weights
    w = 1.0 / (nn_dist ** power)                               # (B, N, k_neighbors)
    w = w / w.sum(dim=-1, keepdim=True)                         # normalize

    # Gather values
    C = sparse_values.shape[-1]
    nn_idx_expanded = nn_idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N, k_neighbors, C)
    nn_vals = torch.gather(
        sparse_values.unsqueeze(1).expand(-1, N, -1, -1),
        2, nn_idx_expanded
    )  # (B, N, k_neighbors, C)

    return (w.unsqueeze(-1) * nn_vals).sum(dim=2)              # (B, N, C)


# ═══════════════════════════════════════════════════════════════════════
# Full Sparse Model
# ═══════════════════════════════════════════════════════════════════════

class SparseModel(nn.Module):
    """
    PhysGTO-Sparse: Learnable sparse sampling + operator learning.

    Args:
        space_size:     spatial dimension (3 for 3D)
        pos_enc_dim:    Fourier encoding frequencies
        cond_dim:       condition vector dimension
        N_block:        number of mixer blocks
        in_dim / out_dim: input/output field channels
        enc_dim:        hidden dimension
        n_head:         attention heads
        n_token:        learnable query tokens
        K_ratio:        fraction of points to sample (0.05 ~ 0.1)
        knn_k:          KNN neighbors for sparse graph
        decoder_mode:   'mlp' or 'cross_attention'
        tau:            Gumbel-Softmax temperature
        dt:             default time step
    """

    def __init__(self,
                 space_size=3,
                 pos_enc_dim=5,
                 cond_dim=32,
                 N_block=4,
                 in_dim=4,
                 out_dim=4,
                 enc_dim=128,
                 n_head=4,
                 n_token=64,
                 K_ratio=0.1,
                 knn_k=16,
                 decoder_mode='cross_attention',
                 tau=1.0,
                 dt: float = 0.05,
                 ):
        super().__init__()

        self.dt = dt
        self.pos_enc_dim = pos_enc_dim
        self.K_ratio = K_ratio
        self.decoder_mode = decoder_mode
        self.space_size = space_size

        # Dimension calculations
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 2 * (1 + 2 * pos_enc_dim)
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.enc_s_dim = enc_s_dim

        # Sampling network: input = state + position
        sampler_feat_dim = in_dim + space_size
        self.sampler = LearnableSampler(
            feat_dim=sampler_feat_dim,
            hidden_dim=enc_dim,
            n_layers=3,
            knn_k=knn_k,
            tau=tau
        )

        # Encoder (operates on K sparse nodes)
        self.encoder = SparseEncoder(
            space_size=space_size,
            state_size=in_dim,
            enc_dim=enc_dim,
            enc_t_dim=enc_t_dim,
            enc_c_dim=enc_c_dim
        )

        # Mixer (GNN + Attention blocks)
        self.mixer = Mixer(
            N=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim
        )

        # Decoder
        if decoder_mode == 'mlp':
            self.decoder = SparseMLPDecoder(
                N_block=N_block,
                enc_dim=enc_dim,
                enc_s_dim=enc_s_dim,
                state_size=out_dim
            )
        elif decoder_mode == 'cross_attention':
            self.decoder = CrossAttentionDecoder(
                N_block=N_block,
                enc_dim=enc_dim,
                enc_s_dim=enc_s_dim,
                state_size=out_dim,
                n_heads=n_head
            )

    def forward(self, state_in, node_pos, edges_unused, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None,
                _sampled_indices=None, _sparse_edges=None, _sampled_pos=None):
        """
        Args:
            state_in:  (B, N, in_dim)  full-grid state
            node_pos:  (B, N, 3)       full-grid positions
            edges_unused: ignored (sparse graph is built internally)
            time_i:    (B,) or (B, 1)  current time
            conditions:(B, cond_dim)   condition vector
            pos_enc:   (B, N, enc_s_dim) precomputed position encoding (optional)
            c_enc:     (B, enc_c_dim) precomputed condition encoding (optional)
            dt:        time step override
            _sampled_indices: (B, K) precomputed indices (for autoregressive reuse)
            _sparse_edges: (B, ne, 2) precomputed KNN edges
            _sampled_pos: (B, K, 3) precomputed sampled positions

        Returns:
            state_pred: (B, N, out_dim) predicted next state on full grid
        """
        B, N, C = state_in.shape

        # ── Precompute encodings if needed ──
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)   # (B, N, enc_s_dim)
            c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)   # (B, enc_c_dim)

        # ── Time encoding ──
        if len(time_i.shape) == 1:
            time_i = time_i.view(-1, 1)
        bs = time_i.shape[0]

        if dt is None:
            dt_val = torch.full((bs, 1), self.dt, dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (float, int)):
            dt_val = torch.full((bs, 1), float(dt), dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (np.floating, np.integer)):
            dt_val = torch.tensor([dt], dtype=time_i.dtype, device=time_i.device).reshape(bs, 1)
        else:
            dt_val = dt.view(bs, 1).to(dtype=time_i.dtype, device=time_i.device)

        time_info = torch.cat([time_i, dt_val], dim=-1)
        t_enc = FourierEmbedding(time_info, 0, self.pos_enc_dim)       # (B, enc_t_dim)

        # ── Sparse Sampling ──
        K = max(1, int(N * self.K_ratio))
        hard = not self.training

        if _sampled_indices is not None:
            # Reuse sampling from autoregressive (indices are fixed per rollout)
            sampled_state = gather_by_indices(state_in, _sampled_indices)   # (B, K, C)
            sampled_pos = _sampled_pos                                       # (B, K, 3)
            sparse_edges = _sparse_edges                                     # (B, ne, 2)
            indices = _sampled_indices
        else:
            sampler_input = torch.cat([state_in, node_pos], dim=-1)        # (B, N, C+3)
            sampled_state, sampled_pos, sparse_edges, weights, indices = \
                self.sampler(sampler_input, node_pos, K, hard=hard)

        # ── Fourier encoding on sampled positions ──
        pos_enc_K = FourierEmbedding(sampled_pos, 0, self.pos_enc_dim)   # (B, K, enc_s_dim)

        # ── Encoder on sparse graph ──
        V, E = self.encoder(sampled_pos, sampled_state, t_enc, c_enc, sparse_edges)
        # V: (B, K, enc_dim), E: (B, ne, enc_dim)

        # ── Mixer ──
        V_all = self.mixer(V, E, sparse_edges, pos_enc_K)
        # V_all: (B, N_block, K, enc_dim)

        # ── Decoder ──
        if self.decoder_mode == 'mlp':
            # Predict delta on K sparse points
            delta_K = self.decoder(V_all, pos_enc_K)                     # (B, K, out_dim)
            # Interpolate to full grid
            delta_full = idw_interpolate(delta_K, sampled_pos, node_pos)  # (B, N, out_dim)
        elif self.decoder_mode == 'cross_attention':
            # Directly decode on full grid via cross-attention
            delta_full = self.decoder(V_all, pos_enc_K, pos_enc)          # (B, N, out_dim)

        state_pred = state_in + delta_full
        return state_pred

    def autoregressive(self,
                       state_in,
                       node_pos,
                       edges,
                       time_seq,
                       conditions,
                       dt=None,
                       check_point=False,
                       teacher_forcing=False,
                       gt_states=None):
        """
        Autoregressive rollout.
        Sampling is done ONCE at the first step and reused for the entire rollout,
        since node positions don't change.
        """
        state_t = state_in
        outputs = [state_in]
        T = time_seq.shape[1]
        B, N, C = state_in.shape

        # Precompute static encodings
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        # Sample ONCE: determine which K points to track
        K = max(1, int(N * self.K_ratio))
        sampler_input = torch.cat([state_in, node_pos], dim=-1)
        _, sampled_pos, sparse_edges, _, indices = \
            self.sampler(sampler_input, node_pos, K, hard=True)

        for t in range(T):
            time_i = time_seq[:, t]

            def custom_forward(s_t, t_i):
                return self.forward(
                    s_t, node_pos, edges, t_i, conditions,
                    pos_enc, c_enc, dt,
                    _sampled_indices=indices,
                    _sparse_edges=sparse_edges,
                    _sampled_pos=sampled_pos
                )

            if check_point:
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                state_pred = checkpoint(custom_forward, state_t, time_i, use_reentrant=False)
            else:
                state_pred = self.forward(
                    state_t, node_pos, edges, time_i, conditions,
                    pos_enc, c_enc, dt,
                    _sampled_indices=indices,
                    _sparse_edges=sparse_edges,
                    _sampled_pos=sampled_pos
                )

            outputs.append(state_pred)

            if t < T - 1:
                if teacher_forcing and gt_states is not None:
                    state_t = gt_states[:, t]
                else:
                    state_t = state_pred

        return torch.stack(outputs[1:], dim=1)  # (B, T, N, out_dim)


# ═══════════════════════════════════════════════════════════════════════
# Wrapper for compatibility with main.py's `Model` import convention
# ═══════════════════════════════════════════════════════════════════════

Model = SparseModel
