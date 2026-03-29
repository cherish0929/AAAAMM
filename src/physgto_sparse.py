"""
PhysGTO-Sparse v3 — Memory-optimized for full-resolution grids (stride=[1,1,1]).

Key memory optimizations:
  1. Scorer: chunked MLP + gradient checkpointing (never hold (B,N,hidden) fully)
  2. pos_enc: computed lazily per-chunk in decoder, never stored as (B,N,enc_s_dim)
  3. CrossAttentionDecoder: chunked query processing, only (B,chunk,K) attention at a time
  4. IDW interpolation: chunked over N, never creates (B,N,K) distance matrix
  5. Scorer input: pos_enc computed in-place per chunk, not pre-materialized

The GNN/Transformer core operates ONLY on K sparse points (K ≪ N),
so it is inherently memory-efficient. The bottleneck was always the
N-dependent operations: scoring, Fourier encoding, and decoding.

Architecture (same data flow as v2, different memory profile):
  state_in (B,N,C) → Sampler(chunked) → (B,K,C) → Encoder → Mixer → Decoder(chunked) → (B,N,C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .sparse_sampling import (
    LearnableSampler, gather_by_indices, soft_gather, build_knn_graph
)


# ═══════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════

def get_edge_info(edges, node_pos):
    """edges: (B, ne, 2), node_pos: (B, K, 3) -> (B, ne, 7)"""
    s = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    r = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = r - s
    return torch.cat([d, -d, (d ** 2).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)], dim=-1)


def FourierEmbedding(pos, pos_start, pos_length):
    """pos: (..., D) -> (..., D*(2*L) + D)"""
    shape = pos.shape
    flat = pos.reshape(-1, shape[-1])
    freq = 2 ** torch.arange(pos_start, pos_start + pos_length,
                             device=pos.device, dtype=torch.float) * torch.pi
    x = flat.unsqueeze(-1) * freq.view(1, 1, -1)  # (-1, D, L)
    emb = torch.cat([x.cos(), x.sin()], dim=-1).view(*shape[:-1], -1)
    return torch.cat([emb, pos], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, layer_norm=True,
                 n_hidden=1, hidden_size=128, act='SiLU'):
        super().__init__()
        acts = {'GELU': nn.GELU(), 'SiLU': nn.SiLU(), 'PReLU': nn.PReLU()}
        self.act = acts[act]
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), self.act]
            for _ in range(1, n_hidden):
                f += [nn.Linear(hidden_size, hidden_size), self.act]
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))
        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


# ═══════════════════════════════════════════════════════════════════════
# Spatial-aware Attention
# ═══════════════════════════════════════════════════════════════════════

class SpatialAtten(nn.Module):
    """Attention with gated spatial position injection."""

    def __init__(self, n_token=128, c_dim=128, n_heads=4, enc_s_dim=39):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(n_token, c_dim))
        self.attention1 = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)
        self.attention3 = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)
        self.pos_proj = nn.Linear(enc_s_dim, c_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, W0, pos_enc):
        W0_s = W0 + torch.sigmoid(self.gate) * self.pos_proj(pos_enc)
        B = W0_s.shape[0]
        Q = self.Q.unsqueeze(0).expand(B, -1, -1)
        W, _ = self.attention1(Q, W0_s, W0_s)
        W, _ = self.attention2(W, W, W)
        W, _ = self.attention3(W0_s, W, W)
        return W


# ═══════════════════════════════════════════════════════════════════════
# GNN on sparse graph
# ═══════════════════════════════════════════════════════════════════════

class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128,
                 output_size=None, layer_norm=False):
        super().__init__()
        output_size = output_size or node_size
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden,
                          layer_norm=layer_norm, act='SiLU', output_size=edge_size)
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden,
                          layer_norm=layer_norm, act='SiLU', output_size=output_size)

    def forward(self, V, E, edges):
        B, K, _ = V.shape
        s = torch.gather(V, 1, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        r = torch.gather(V, 1, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        ee = self.f_edge(torch.cat([s, r, E], dim=-1))
        e0, e1 = ee.chunk(2, dim=-1)
        a0 = scatter_mean(e0, edges[..., 0].unsqueeze(-1).expand(-1, -1, e0.shape[-1]), dim=1, dim_size=K)
        a1 = scatter_mean(e1, edges[..., 1].unsqueeze(-1).expand(-1, -1, e1.shape[-1]), dim=1, dim_size=K)
        v = self.f_node(torch.cat([V, a0, a1], dim=-1))
        return v, ee


# ═══════════════════════════════════════════════════════════════════════
# MixerBlock & Mixer
# ═══════════════════════════════════════════════════════════════════════

class MixerBlock(nn.Module):
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        self.gnn = GNN(node_size=enc_dim + enc_s_dim, edge_size=enc_dim,
                        output_size=enc_dim, layer_norm=True)
        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.mha = SpatialAtten(n_token, enc_dim, n_head, enc_s_dim)
        self.ffn = nn.Sequential(nn.Linear(enc_dim, 2*enc_dim), nn.SiLU(),
                                 nn.Linear(2*enc_dim, enc_dim))

    def forward(self, V, E, edges, s_enc):
        v, e = self.gnn(torch.cat([V, s_enc], dim=-1), E, edges)
        E = E + e; V = V + v
        V = V + self.mha(self.ln1(V), s_enc)
        V = V + self.ffn(self.ln2(V))
        return V, E


class Mixer(nn.Module):
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock(enc_dim, n_head, n_token, enc_s_dim) for _ in range(N)
        ])

    def forward(self, V, E, edges, pos_enc):
        V_all = []
        for blk in self.blocks:
            V, E = blk(V, E, edges, pos_enc)
            V_all.append(V)
        return torch.stack(V_all, dim=1)


# ═══════════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════════

class SparseEncoder(nn.Module):
    def __init__(self, space_size=3, state_size=4, enc_dim=128,
                 enc_t_dim=11, enc_c_dim=12):
        super().__init__()
        self.fv1 = MLP(input_size=state_size + space_size, output_size=enc_dim,
                       act='SiLU', layer_norm=False)
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim,
                           act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim,
                           act='SiLU', layer_norm=False)
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim,
                      n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        V = self.fv1(torch.cat([state_in, node_pos], -1)) + \
            self.fv_time(time_i).unsqueeze(-2) + self.fv_cond(conditions).unsqueeze(-2)
        E = self.fe(get_edge_info(edges, node_pos))
        return V, E


# ═══════════════════════════════════════════════════════════════════════
# Decoder A: MLP (on K sparse points)
# ═══════════════════════════════════════════════════════════════════════

class SparseMLPDecoder(nn.Module):
    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10, state_size=1):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(N_block * enc_dim + enc_s_dim, enc_dim), nn.SiLU(),
            nn.Linear(enc_dim, enc_dim), nn.SiLU(),
            nn.Linear(enc_dim, state_size))

    def forward(self, V_all, pos_enc):
        B, Nb, K, d = V_all.shape
        V = V_all.permute(0, 2, 1, 3).reshape(B, K, -1)
        return self.delta_net(torch.cat([V, pos_enc], -1))


# ═══════════════════════════════════════════════════════════════════════
# Decoder B: Cross-Attention — CHUNKED over full grid queries
# ═══════════════════════════════════════════════════════════════════════

class CrossAttentionDecoder(nn.Module):
    """
    Memory-safe cross-attention decoder.

    Instead of computing attention over all N query points at once
    (which creates a (B, heads, N, K) matrix — catastrophic for large N),
    we process queries in chunks of `query_chunk_size`.

    Peak attention memory: (B, heads, chunk_size, K) — controllable.
    """

    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10,
                 state_size=1, n_heads=4, query_chunk_size=50000):
        super().__init__()
        self.query_chunk_size = query_chunk_size
        self.n_heads = n_heads
        self.enc_dim = enc_dim

        self.latent_proj = nn.Sequential(
            nn.Linear(N_block * enc_dim, enc_dim), nn.SiLU())
        self.query_proj = nn.Linear(enc_s_dim, enc_dim)
        self.kv_proj = nn.Linear(enc_dim + enc_s_dim, enc_dim * 2)

        self.ln1 = nn.LayerNorm(enc_dim)
        self.out_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim), nn.SiLU(),
            nn.Linear(enc_dim, state_size))

    def forward(self, V_all, pos_enc_K, full_node_pos, pos_enc_dim):
        """
        V_all:         (B, N_block, K, enc_dim)
        pos_enc_K:     (B, K, enc_s_dim)
        full_node_pos: (B, N, 3)  — raw positions, NOT pre-encoded
        pos_enc_dim:   int        — for lazy FourierEmbedding per chunk

        Returns: (B, N, out_dim)
        """
        B, Nb, K, d = V_all.shape
        N = full_node_pos.shape[1]

        # Prepare Key/Value from sparse latent (small — K points)
        V_flat = V_all.permute(0, 2, 1, 3).reshape(B, K, -1)  # (B, K, Nb*d)
        V_proj = self.latent_proj(V_flat)                       # (B, K, d)
        kv = self.kv_proj(torch.cat([V_proj, pos_enc_K], -1))   # (B, K, 2d)
        key, value = kv.chunk(2, dim=-1)                         # each (B, K, d)

        # Reshape K/V for manual attention: (B, heads, K, head_dim)
        head_dim = d // self.n_heads
        key = key.view(B, K, self.n_heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(B, K, self.n_heads, head_dim).permute(0, 2, 1, 3)
        scale = head_dim ** -0.5

        # Process full grid queries in chunks
        out_chunks = []
        for start in range(0, N, self.query_chunk_size):
            end = min(start + self.query_chunk_size, N)
            pos_chunk = full_node_pos[:, start:end]              # (B, chunk, 3)

            # Lazy Fourier encoding: compute only for this chunk
            pos_enc_chunk = FourierEmbedding(pos_chunk, 0, pos_enc_dim)  # (B, chunk, enc_s_dim)
            q_chunk = self.query_proj(pos_enc_chunk)                      # (B, chunk, d)

            # Manual scaled-dot-product attention (chunked)
            q = q_chunk.view(B, -1, self.n_heads, head_dim).permute(0, 2, 1, 3)
            # attn: (B, heads, chunk, K) — bounded by chunk_size
            q, key = F.normalize(q, dim=-1), F.normalize(key, dim=-1)

            attn = torch.matmul(q, key.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, value)  # (B, heads, chunk, head_dim)
            out = out.permute(0, 2, 1, 3).reshape(B, -1, d)  # (B, chunk, d)

            # Residual + norm + output head
            out = self.ln1(out + q_chunk)
            out = self.out_head(out)  # (B, chunk, out_dim)
            out_chunks.append(out)

        return torch.cat(out_chunks, dim=1)  # (B, N, out_dim)


# ═══════════════════════════════════════════════════════════════════════
# IDW interpolation — CHUNKED to avoid (N, K) distance matrix
# ═══════════════════════════════════════════════════════════════════════

def idw_interpolate(sparse_values, sparse_pos, full_pos,
                    power=2.0, k_neighbors=8, chunk_size=50000):
    """
    Chunked IDW interpolation.

    Instead of computing (B, N, K) pairwise distances at once,
    processes N in chunks. Peak memory: (B, chunk, K).
    """
    B, N, D = full_pos.shape
    K = sparse_pos.shape[1]
    k_neighbors = min(k_neighbors, K)
    C = sparse_values.shape[-1]

    out_chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        fp = full_pos[:, start:end]  # (B, chunk, 3)
        chunk_n = fp.shape[1]

        # Distance: (B, chunk, K)
        diff = fp.unsqueeze(2) - sparse_pos.unsqueeze(1)
        dist = (diff ** 2).sum(-1).sqrt().clamp(min=1e-8)

        _, nn_idx = dist.topk(k_neighbors, dim=-1, largest=False)  # (B, chunk, kn)
        nn_dist = torch.gather(dist, 2, nn_idx)

        w = 1.0 / (nn_dist ** power)
        w = w / w.sum(dim=-1, keepdim=True)

        nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        nn_vals = torch.gather(
            sparse_values.unsqueeze(1).expand(-1, chunk_n, -1, -1),
            2, nn_idx_exp
        )
        out_chunks.append((w.unsqueeze(-1) * nn_vals).sum(dim=2))

    return torch.cat(out_chunks, dim=1)


# ═══════════════════════════════════════════════════════════════════════
# Chunked scorer input builder
# ═══════════════════════════════════════════════════════════════════════

def _build_scorer_input_and_score(scorer, state_in, node_pos,
                                  t_enc, c_enc, pos_enc_dim,
                                  chunk_size=100000):
    """
    Build enriched scorer input and evaluate in chunks.

    Instead of:
      pos_enc = FourierEmbedding(node_pos)   # (B, N, enc_s_dim) — huge!
      scorer_input = cat([state, pos_enc, t, c])  # (B, N, big) — huge!
      logits = scorer(scorer_input)           # 3-layer MLP on all N — huge activations!

    We compute Fourier encoding and MLP per chunk, so peak memory is:
      (B, chunk, scorer_feat_dim) + (B, chunk, hidden_dim)

    Returns: logits (B, N)
    """
    B, N, C = state_in.shape
    t_bc = t_enc.unsqueeze(1)  # (B, 1, enc_t_dim) — will broadcast
    c_bc = c_enc.unsqueeze(1)  # (B, 1, enc_c_dim)

    logits_chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        # Lazy Fourier encoding: only for this chunk of positions
        pos_chunk = node_pos[:, start:end]             # (B, chunk, 3)
        pos_enc_chunk = FourierEmbedding(pos_chunk, 0, pos_enc_dim)  # (B, chunk, enc_s_dim)

        state_chunk = state_in[:, start:end]            # (B, chunk, C)
        t_chunk = t_bc.expand(-1, end - start, -1)      # (B, chunk, enc_t_dim)
        c_chunk = c_bc.expand(-1, end - start, -1)      # (B, chunk, enc_c_dim)

        scorer_in = torch.cat([state_chunk, pos_enc_chunk, t_chunk, c_chunk], dim=-1)
        logit_chunk = scorer.scorer(scorer_in, chunk_size=0)  # already small, no inner chunk
        logits_chunks.append(logit_chunk)                      # (B, chunk)

    return torch.cat(logits_chunks, dim=1)  # (B, N)


# ═══════════════════════════════════════════════════════════════════════
# Full Sparse Model — memory-optimized
# ═══════════════════════════════════════════════════════════════════════

class SparseModel(nn.Module):
    """
    PhysGTO-Sparse v3.

    Memory budget at stride=[1,1,1] (N ≈ millions):
      - Full-grid ops (scoring, decoding): chunked, O(chunk) memory
      - Core GNN/Transformer: O(K) memory where K = N * K_ratio
      - No full (B,N,enc_s_dim) tensor is ever materialized

    New params:
      scorer_chunk_size:  chunk size for full-grid scoring MLP
      decoder_chunk_size: chunk size for cross-attention / IDW decoding
      radius_cutoff:      distance threshold for KNN pruning
      resample_every:     resample frequency in autoregressive rollout
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
                 radius_cutoff=0.0,
                 decoder_mode='cross_attention',
                 tau=1.0,
                 resample_every=5,
                 scorer_chunk_size=100000,
                 decoder_chunk_size=50000,
                 dt: float = 0.05,
                 ):
        super().__init__()

        self.dt = dt
        self.pos_enc_dim = pos_enc_dim
        self.K_ratio = K_ratio
        self.decoder_mode = decoder_mode
        self.space_size = space_size
        self.resample_every = resample_every
        self.scorer_chunk_size = scorer_chunk_size
        self.decoder_chunk_size = decoder_chunk_size

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 2 * (1 + 2 * pos_enc_dim)
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.enc_s_dim = enc_s_dim
        self.enc_t_dim = enc_t_dim
        self.enc_c_dim = enc_c_dim

        # Scorer input dim (used for weight init; actual computation is chunked)
        scorer_feat_dim = in_dim + enc_s_dim + enc_t_dim + enc_c_dim
        self.sampler = LearnableSampler(
            feat_dim=scorer_feat_dim, hidden_dim=enc_dim, n_layers=3,
            knn_k=knn_k, radius_cutoff=radius_cutoff, tau=tau,
            scorer_chunk_size=scorer_chunk_size)

        self.encoder = SparseEncoder(
            space_size, in_dim, enc_dim, enc_t_dim, enc_c_dim)

        self.mixer = Mixer(N_block, enc_dim, n_head, n_token, enc_s_dim)

        if decoder_mode == 'mlp':
            self.decoder = SparseMLPDecoder(N_block, enc_dim, enc_s_dim, out_dim)
        elif decoder_mode == 'cross_attention':
            self.decoder = CrossAttentionDecoder(
                N_block, enc_dim, enc_s_dim, out_dim, n_head, decoder_chunk_size)

    def _build_time_encoding(self, time_i, dt, bs, device, dtype):
        if len(time_i.shape) == 1:
            time_i = time_i.view(-1, 1)
        if dt is None:
            dt_v = torch.full((bs, 1), self.dt, dtype=dtype, device=device)
        elif isinstance(dt, (float, int)):
            dt_v = torch.full((bs, 1), float(dt), dtype=dtype, device=device)
        elif isinstance(dt, (np.floating, np.integer)):
            dt_v = torch.tensor([float(dt)], dtype=dtype, device=device).expand(bs, 1)
        else:
            dt_v = dt.view(bs, 1).to(dtype=dtype, device=device)
        return FourierEmbedding(torch.cat([time_i, dt_v], -1), 0, self.pos_enc_dim)

    def _do_sampling(self, state_in, node_pos, t_enc, c_enc, hard):
        """
        Chunked scoring + sampling.
        pos_enc is computed per-chunk inside _build_scorer_input_and_score,
        never materialized as a full (B, N, enc_s_dim) tensor.
        """
        from .sparse_sampling import soft_topk, soft_gather, gather_by_indices, build_knn_graph

        B, N, C = state_in.shape
        K = max(1, int(N * self.K_ratio))

        # Chunked scoring: never hold (B, N, hidden) or (B, N, enc_s_dim) fully
        logits = _build_scorer_input_and_score(
            self.sampler, state_in, node_pos, t_enc, c_enc,
            self.pos_enc_dim, self.scorer_chunk_size)  # (B, N)

        weights_K, indices = soft_topk(logits, K, self.sampler.tau, hard)

        node_features = torch.cat([state_in, node_pos], dim=-1)
        sampled_features = soft_gather(node_features, indices, weights_K)
        sampled_pos = gather_by_indices(node_pos, indices)
        edges = build_knn_graph(sampled_pos, self.sampler.knn_k, self.sampler.radius_cutoff)

        return sampled_features, sampled_pos, edges, weights_K, indices

    def forward(self, state_in, node_pos, edges_unused, time_i, conditions,
                pos_enc_unused=None, c_enc=None, dt=None,
                _sampled_indices=None, _sparse_edges=None,
                _sampled_pos=None, _weights_K=None):
        """
        Memory-optimized forward.

        NOTE: pos_enc is NOT pre-computed or stored for the full grid.
        The decoder computes Fourier encoding lazily per chunk.
        c_enc is still precomputed (it's only (B, enc_c_dim) — tiny).
        """
        B, N, C = state_in.shape

        if c_enc is None:
            c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        t_enc = self._build_time_encoding(time_i, dt, B, state_in.device, state_in.dtype)

        # ── Sampling (chunked if from scratch) ──
        if _sampled_indices is not None:
            if _weights_K is not None:
                sf = soft_gather(torch.cat([state_in, node_pos], -1),
                                 _sampled_indices, _weights_K)
            else:
                sf = gather_by_indices(torch.cat([state_in, node_pos], -1),
                                       _sampled_indices)
            sampled_pos, sparse_edges = _sampled_pos, _sparse_edges
        else:
            hard = not self.training
            sf, sampled_pos, sparse_edges, _, _ = \
                self._do_sampling(state_in, node_pos, t_enc, c_enc, hard)

        sampled_state = sf[..., :C]
        # pos_enc on K points only — small
        pos_enc_K = FourierEmbedding(sampled_pos, 0, self.pos_enc_dim)

        # ── Core: Encoder → Mixer (all on K sparse points — memory-efficient) ──
        V, E = self.encoder(sampled_pos, sampled_state, t_enc, c_enc, sparse_edges)
        V_all = self.mixer(V, E, sparse_edges, pos_enc_K)

        # ── Decoder (chunked over full grid N) ──
        if self.decoder_mode == 'mlp':
            delta_K = self.decoder(V_all, pos_enc_K)  # (B, K, out_dim)
            delta_full = idw_interpolate(
                delta_K, sampled_pos, node_pos,
                chunk_size=self.decoder_chunk_size)
        elif self.decoder_mode == 'cross_attention':
            # Pass raw node_pos, NOT pos_enc — decoder computes Fourier per chunk
            delta_full = self.decoder(V_all, pos_enc_K, node_pos, self.pos_enc_dim)

        return state_in + delta_full

    def autoregressive(self, state_in, node_pos, edges, time_seq, conditions,
                       dt=None, check_point=False,
                       teacher_forcing=False, gt_states=None):
        """
        Autoregressive rollout with periodic resampling.
        """
        state_t = state_in
        outputs = [state_in]
        T = time_seq.shape[1]
        B, N, C = state_in.shape

        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        cached_indices = cached_pos = cached_edges = None

        for t in range(T):
            time_i = time_seq[:, t]

            if t % self.resample_every == 0 or cached_indices is None:
                t_enc = self._build_time_encoding(
                    time_i, dt, B, state_t.device, state_t.dtype)
                _, cached_pos, cached_edges, _, cached_indices = \
                    self._do_sampling(state_t, node_pos, t_enc, c_enc, hard=True)

            def custom_forward(s_t, t_i):
                return self.forward(
                    s_t, node_pos, edges, t_i, conditions, None, c_enc, dt,
                    cached_indices, cached_edges, cached_pos, None)

            if check_point:
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                state_pred = grad_checkpoint(custom_forward, state_t, time_i,
                                             use_reentrant=False)
            else:
                state_pred = custom_forward(state_t, time_i)

            outputs.append(state_pred)
            if t < T - 1:
                state_t = gt_states[:, t] if (teacher_forcing and gt_states is not None) \
                          else state_pred

        return torch.stack(outputs[1:], dim=1)


Model = SparseModel
