"""
PhysGTO with Attention Residuals (AttnRes).

Based on: "Attention Residuals" (Kimi Team, arXiv:2603.15031)
Two modes controlled by `residual_mode`:
  - "full_attnres": every sub-layer is an individual source in depth-wise attention
  - "block_attnres": layers grouped into blocks; intra-block standard residuals,
                     inter-block attention residuals over block-level representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from torch_scatter import scatter_mean, scatter_softmax

VELOCITY_FIELD_NAMES = ("Ux", "Uy", "Uz")

# ============================================================
# Shared utilities (identical to physgto.py)
# ============================================================

def get_edge_info(edges, node_pos):
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True))
    E = torch.cat([d, -d, norm], dim=-1)
    return E

def broadcast_dt(dt, ref_tensor):
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, dtype=ref_tensor.dtype, device=ref_tensor.device)
    else:
        dt = dt.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    if dt.dim() == 0:
        dt = dt.view(1, 1, 1)
    elif dt.dim() == 1:
        dt = dt.view(-1, 1, 1)
    elif dt.dim() == 2:
        dt = dt.unsqueeze(-1)
    return dt

class MLP(nn.Module):
    def __init__(self,
                input_size = 128,
                output_size = 128,
                layer_norm = True,
                n_hidden=1,
                hidden_size = 128,
                act = 'SiLU',
                dropout=0.0,
                ):
        super(MLP, self).__init__()
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
            if dropout > 0:
                f.append(nn.Dropout(dropout))
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
                if dropout > 0:
                    f.append(nn.Dropout(dropout))
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class Atten(nn.Module):
    """
    Multi-scale Perceiver-style attention.
    Coarse tokens capture global patterns (pressure-like), fine tokens capture local gradients.
    """
    def __init__(self,
                n_token=128,
                c_dim=128,
                n_heads=4):
        super(Atten, self).__init__()

        self.c_dim = c_dim
        self.n_token = n_token
        self.n_heads = n_heads

        n_coarse = max(n_token // 4, 4)
        n_fine = n_token

        self.Q_coarse = nn.Parameter(torch.randn(n_coarse, c_dim) * 0.02, requires_grad=True)
        self.Q_fine = nn.Parameter(torch.randn(n_fine, c_dim) * 0.02, requires_grad=True)

        self.attn_compress_coarse = nn.MultiheadAttention(embed_dim=c_dim, num_heads=n_heads, batch_first=True)
        self.attn_compress_fine = nn.MultiheadAttention(embed_dim=c_dim, num_heads=n_heads, batch_first=True)

        self.attn_self = nn.MultiheadAttention(embed_dim=c_dim, num_heads=n_heads, batch_first=True)

        self.attn_decompress = nn.MultiheadAttention(embed_dim=c_dim, num_heads=n_heads, batch_first=True)

        self.merge = nn.Sequential(
            nn.Linear(c_dim * 2, c_dim),
            nn.SiLU(),
            nn.Linear(c_dim, c_dim),
        )

    def forward(self, W0):
        batch = W0.shape[0]

        Q_c = self.Q_coarse.unsqueeze(0).expand(batch, -1, -1)
        W_coarse, _ = self.attn_compress_coarse(Q_c, W0, W0)

        Q_f = self.Q_fine.unsqueeze(0).expand(batch, -1, -1)
        W_fine, _ = self.attn_compress_fine(Q_f, W0, W0)

        W_cat = torch.cat([W_coarse, W_fine], dim=1)
        W_cat, _ = self.attn_self(W_cat, W_cat, W_cat)

        n_c = W_coarse.shape[1]
        W_coarse_out = W_cat[:, :n_c]
        W_fine_out = W_cat[:, n_c:]

        out_coarse, _ = self.attn_decompress(W0, W_coarse_out, W_coarse_out)
        out_fine, _ = self.attn_decompress(W0, W_fine_out, W_fine_out)

        out = self.merge(torch.cat([out_coarse, out_fine], dim=-1))

        return out


def FourierEmbedding(pos, pos_start, pos_length):
    original_shape = pos.shape
    new_pos = pos.reshape(-1, original_shape[-1])
    index = torch.arange(pos_start, pos_start + pos_length, device=pos.device)
    index = index.float()
    freq = 2 ** index * torch.pi
    cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    embedding = torch.cat([cos_feat, sin_feat], dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    all_embeddings = torch.cat([embedding, pos], dim=-1)
    return all_embeddings


class GatedGNN(nn.Module):
    """
    GNN with edge-gated attention aggregation and separate direction scoring.
    """
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GatedGNN, self).__init__()

        self.node_size = node_size
        self.output_size = output_size
        self.edge_size = edge_size
        output_size = output_size or node_size

        self.f_edge = MLP(
            input_size=edge_size + node_size * 2,
            n_hidden=n_hidden,
            layer_norm=layer_norm,
            act='SiLU',
            output_size=edge_size
        )

        self.f_attn_0 = nn.Sequential(
            nn.Linear(edge_size // 2, edge_size // 4),
            nn.SiLU(),
            nn.Linear(edge_size // 4, 1),
        )
        self.f_attn_1 = nn.Sequential(
            nn.Linear(edge_size // 2, edge_size // 4),
            nn.SiLU(),
            nn.Linear(edge_size // 4, 1),
        )

        self.f_node = MLP(
            input_size=edge_size + node_size,
            n_hidden=n_hidden,
            layer_norm=layer_norm,
            act='SiLU',
            output_size=output_size
        )

    def forward(self, V, E, edges):
        bs, N, _ = V.shape

        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        edge_emb_0, edge_emb_1 = edge_embeddings.chunk(2, dim=-1)
        feat_dim = edge_emb_0.shape[-1]

        attn_logits_0 = self.f_attn_0(edge_emb_0).squeeze(-1)
        attn_logits_1 = self.f_attn_1(edge_emb_1).squeeze(-1)

        recv_idx_0 = edges[..., 0]
        recv_idx_1 = edges[..., 1]

        attn_weights_0 = scatter_softmax(attn_logits_0, recv_idx_0, dim=1, dim_size=N)
        attn_weights_1 = scatter_softmax(attn_logits_1, recv_idx_1, dim=1, dim_size=N)

        weighted_msg_0 = edge_emb_0 * attn_weights_0.unsqueeze(-1)
        weighted_msg_1 = edge_emb_1 * attn_weights_1.unsqueeze(-1)

        col_0 = recv_idx_0.unsqueeze(-1).expand(-1, -1, feat_dim)
        col_1 = recv_idx_1.unsqueeze(-1).expand(-1, -1, feat_dim)

        agg_0 = torch.zeros(bs, N, feat_dim, device=V.device, dtype=V.dtype)
        agg_1 = torch.zeros(bs, N, feat_dim, device=V.device, dtype=V.dtype)
        agg_0.scatter_add_(1, col_0, weighted_msg_0)
        agg_1.scatter_add_(1, col_1, weighted_msg_1)

        edge_mean = torch.cat([agg_0, agg_1], dim=-1)
        node_inpt = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


# ============================================================
# NS-Decomposed Decoder (identical to physgto.py)
# ============================================================

class OperatorHead(nn.Module):
    """A lightweight per-component head that maps (enc_dim) -> (state_size)."""
    def __init__(self, enc_dim, state_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.SiLU(),
            nn.Linear(enc_dim // 2, state_size),
        )
    def forward(self, h):
        return self.net(h)


class NSDecomposedDecoder(nn.Module):
    """
    Decoder with NS-inspired operator decomposition.
    """
    def __init__(self, N=4, enc_dim=128, enc_s_dim=10, state_size=3):
        super().__init__()
        self.state_size = state_size
        self.N = N
        self.enc_dim = enc_dim
        in_dim = N * enc_dim + enc_s_dim

        self.proj = nn.Linear(in_dim, enc_dim)
        self.proj_norm = nn.LayerNorm(enc_dim)

        self.diff_proj = nn.Linear(enc_dim + enc_s_dim, enc_dim)
        self.diff_res = nn.Sequential(
            nn.LayerNorm(enc_dim), nn.Linear(enc_dim, enc_dim), nn.SiLU(), nn.Linear(enc_dim, enc_dim),
        )
        self.diff_head = OperatorHead(enc_dim, state_size)

        self.conv_res = nn.Sequential(
            nn.LayerNorm(enc_dim), nn.Linear(enc_dim, enc_dim * 2), nn.SiLU(), nn.Linear(enc_dim * 2, enc_dim),
        )
        self.conv_gate = nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.Sigmoid())
        self.conv_head = OperatorHead(enc_dim, state_size)

        self.press_proj = nn.Linear(enc_dim + enc_s_dim, enc_dim)
        self.press_res = nn.Sequential(
            nn.LayerNorm(enc_dim), nn.Linear(enc_dim, enc_dim), nn.SiLU(), nn.Linear(enc_dim, enc_dim),
        )
        self.press_head = OperatorHead(enc_dim, state_size)

        self.src_res1 = nn.Sequential(
            nn.LayerNorm(enc_dim), nn.Linear(enc_dim, enc_dim * 2), nn.SiLU(), nn.Linear(enc_dim * 2, enc_dim),
        )
        self.src_gate1 = nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.Sigmoid())
        self.src_res2 = nn.Sequential(
            nn.LayerNorm(enc_dim), nn.Linear(enc_dim, enc_dim), nn.SiLU(), nn.Linear(enc_dim, enc_dim),
        )
        self.src_gate2 = nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.Sigmoid())
        self.src_head = OperatorHead(enc_dim, state_size)

        self.combiner = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2), nn.SiLU(), nn.Linear(enc_dim // 2, 4),
        )

    def forward(self, V_all, pos_enc):
        b, n_block, N, enc_dim = V_all.shape

        V_first = V_all[:, 0]
        V_last = V_all[:, -1]

        V_cat = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        h_full = self.proj_norm(self.proj(torch.cat([V_cat, pos_enc], dim=-1)))

        h_diff = self.diff_proj(torch.cat([V_last, pos_enc], dim=-1))
        h_diff = h_diff + self.diff_res(h_diff)
        delta_diff = self.diff_head(h_diff)

        r_conv = self.conv_res(h_full)
        g_conv = self.conv_gate(h_full)
        h_conv = h_full + g_conv * r_conv
        delta_conv = self.conv_head(h_conv)

        h_press = self.press_proj(torch.cat([V_first, pos_enc], dim=-1))
        h_press = h_press + self.press_res(h_press)
        delta_press = self.press_head(h_press)

        h_src = h_full
        h_src = h_src + self.src_gate1(h_src) * self.src_res1(h_src)
        h_src = h_src + self.src_gate2(h_src) * self.src_res2(h_src)
        delta_src = self.src_head(h_src)

        weights = F.softmax(self.combiner(h_full), dim=-1)
        deltas = torch.stack([delta_diff, delta_conv, delta_press, delta_src], dim=2)
        delta = (weights.unsqueeze(-1) * deltas).sum(dim=2)

        return delta


class Encoder(nn.Module):
    """Enhanced encoder with FiLM conditioning."""
    def __init__(self,
                 space_size=2,
                 state_size=4,
                 enc_dim=128,
                 enc_t_dim=11,
                 enc_c_dim=12
                 ):
        super(Encoder, self).__init__()

        self.fv1 = MLP(input_size=state_size + space_size + 1 + state_size, output_size=enc_dim, act='SiLU', layer_norm=False)

        self.film_time = nn.Sequential(
            nn.Linear(enc_t_dim, enc_dim), nn.SiLU(), nn.Linear(enc_dim, enc_dim * 2),
        )
        self.film_cond = nn.Sequential(
            nn.Linear(enc_c_dim, enc_dim), nn.SiLU(), nn.Linear(enc_dim, enc_dim * 2),
        )

        self.fe = MLP(input_size=2 * space_size + 1 + state_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        vel_mag = torch.norm(state_in, dim=-1, keepdim=True)
        vel_sign = torch.sign(state_in)

        state_aug = torch.cat((state_in, node_pos, vel_mag, vel_sign), dim=-1)
        V = self.fv1(state_aug)

        time_film = self.film_time(time_i)
        t_gamma, t_beta = time_film.chunk(2, dim=-1)
        t_gamma = torch.tanh(t_gamma) * 0.5
        V = V * (1.0 + t_gamma.unsqueeze(-2)) + t_beta.unsqueeze(-2)

        cond_film = self.film_cond(conditions)
        c_gamma, c_beta = cond_film.chunk(2, dim=-1)
        c_gamma = torch.tanh(c_gamma) * 0.5
        V = V * (1.0 + c_gamma.unsqueeze(-2)) + c_beta.unsqueeze(-2)

        spatial_edge = get_edge_info(edges, node_pos)

        send_vel = torch.gather(state_in, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, state_in.shape[-1]))
        recv_vel = torch.gather(state_in, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, state_in.shape[-1]))
        vel_diff = recv_vel - send_vel

        send_mag = torch.norm(send_vel, dim=-1, keepdim=True)
        recv_mag = torch.norm(recv_vel, dim=-1, keepdim=True)
        mag_diff = recv_mag - send_mag

        edge_input = torch.cat([spatial_edge, vel_diff, mag_diff], dim=-1)
        E = self.fe(edge_input)

        return V, E


# ============================================================
# Attention Residuals core
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm without learnable scale (used for AttnRes keys)."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def attn_res_op(sources, query_weight, norm):
    """
    Core Attention Residuals operation.

    Args:
        sources: list of tensors [B, N, d] — all available source representations
        query_weight: nn.Parameter [d] — learned pseudo-query for this position
        norm: RMSNorm instance for key normalization

    Returns:
        h: [B, N, d] — weighted combination of sources via softmax attention
    """
    V = torch.stack(sources, dim=0)           # [S, B, N, d]
    K = norm(V)                                # [S, B, N, d]
    logits = torch.einsum('s b n d, d -> s b n', K, query_weight)  # [S, B, N]
    weights = torch.softmax(logits, dim=0)     # [S, B, N]
    h = torch.einsum('s b n, s b n d -> b n d', weights, V)        # [B, N, d]
    return h


# ============================================================
# Block AttnRes: MixerBlock with inter-block attention residuals
# ============================================================

class MixerBlockAttnRes(nn.Module):
    """
    MixerBlock with Block Attention Residuals.

    Intra-block: standard additive residuals (V = V + alpha * f(V))
    Inter-block: before each sub-layer, compute AttnRes over previous block
                 representations and current intra-block partial sum.
    """
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        node_size = enc_dim + enc_s_dim

        self.gnn = GatedGNN(
            node_size=node_size,
            edge_size=enc_dim,
            output_size=enc_dim,
            layer_norm=True
        )

        self.ln0 = nn.LayerNorm(enc_dim)
        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.mha = Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head)

        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, 2 * enc_dim),
            nn.SiLU(),
            nn.Linear(2 * enc_dim, enc_dim)
        )

        # Learnable residual scaling for intra-block
        self.alpha_gnn = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_attn = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_ffn = nn.Parameter(torch.ones(1) * 0.5)

        # AttnRes pseudo-queries — one per sub-layer, initialized to ZERO
        self.w_gnn = nn.Parameter(torch.zeros(enc_dim))
        self.w_attn = nn.Parameter(torch.zeros(enc_dim))
        self.w_ffn = nn.Parameter(torch.zeros(enc_dim))

        # Shared RMSNorm for AttnRes keys
        self.attn_res_norm = RMSNorm(enc_dim)

    def forward(self, blocks, partial_block, E, edges, s_enc):
        """
        Args:
            blocks: list of [B, N, d] — completed block representations
            partial_block: [B, N, d] or None — current intra-block partial sum
            E: [B, ne, d] — edge embeddings
            edges: [B, ne, 2] — edge indices
            s_enc: [B, N, s_dim] — positional encoding
        Returns:
            blocks, partial_block, E
        """
        # --- AttnRes before GNN ---
        sources = blocks + ([partial_block] if partial_block is not None else [])
        h = attn_res_op(sources, self.w_gnn, self.attn_res_norm)

        # GNN sub-layer (standard intra-block residual)
        V_in = torch.cat([self.ln0(h), s_enc], dim=-1)
        v_gnn, e = self.gnn(V_in, E, edges)
        E = E + e
        if partial_block is not None:
            partial_block = partial_block + self.alpha_gnn * v_gnn
        else:
            partial_block = v_gnn

        # --- AttnRes before Atten ---
        sources = blocks + [partial_block]
        h = attn_res_op(sources, self.w_attn, self.attn_res_norm)

        # Atten sub-layer
        v_attn = self.mha(self.ln1(h))
        partial_block = partial_block + self.alpha_attn * v_attn

        # --- AttnRes before FFN ---
        sources = blocks + [partial_block]
        h = attn_res_op(sources, self.w_ffn, self.attn_res_norm)

        # FFN sub-layer
        v_ffn = self.ffn(self.ln2(h))
        partial_block = partial_block + self.alpha_ffn * v_ffn

        return blocks, partial_block, E


# ============================================================
# Full AttnRes: MixerBlock where every sub-layer is a source
# ============================================================

class MixerBlockFullAttnRes(nn.Module):
    """
    MixerBlock with Full Attention Residuals.

    Every sub-layer output becomes an individual source for future layers.
    No standard additive residuals — AttnRes replaces them entirely.
    """
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        node_size = enc_dim + enc_s_dim

        self.gnn = GatedGNN(
            node_size=node_size,
            edge_size=enc_dim,
            output_size=enc_dim,
            layer_norm=True
        )

        self.ln0 = nn.LayerNorm(enc_dim)
        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.mha = Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head)

        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, 2 * enc_dim),
            nn.SiLU(),
            nn.Linear(2 * enc_dim, enc_dim)
        )

        # AttnRes pseudo-queries — one per sub-layer, initialized to ZERO
        self.w_gnn = nn.Parameter(torch.zeros(enc_dim))
        self.w_attn = nn.Parameter(torch.zeros(enc_dim))
        self.w_ffn = nn.Parameter(torch.zeros(enc_dim))

        # Shared RMSNorm for AttnRes keys
        self.attn_res_norm = RMSNorm(enc_dim)

    def forward(self, sources, E, edges, s_enc):
        """
        Args:
            sources: list of [B, N, d] — all previous layer/embedding outputs
            E: [B, ne, d] — edge embeddings
            edges: [B, ne, 2] — edge indices
            s_enc: [B, N, s_dim] — positional encoding
        Returns:
            new_sources (new list, input not mutated), E
        """
        # Build a new list to avoid in-place mutation (required for checkpoint)
        new_sources = list(sources)

        # --- AttnRes before GNN ---
        h = attn_res_op(new_sources, self.w_gnn, self.attn_res_norm)

        # GNN sub-layer
        V_in = torch.cat([self.ln0(h), s_enc], dim=-1)
        v_gnn, e = self.gnn(V_in, E, edges)
        E = E + e
        new_sources.append(v_gnn)

        # --- AttnRes before Atten ---
        h = attn_res_op(new_sources, self.w_attn, self.attn_res_norm)

        # Atten sub-layer
        v_attn = self.mha(self.ln1(h))
        new_sources.append(v_attn)

        # --- AttnRes before FFN ---
        h = attn_res_op(new_sources, self.w_ffn, self.attn_res_norm)

        # FFN sub-layer
        v_ffn = self.ffn(self.ln2(h))
        new_sources.append(v_ffn)

        return new_sources, E


# ============================================================
# Mixer with AttnRes routing
# ============================================================

class MixerAttnRes(nn.Module):
    """
    Mixer that supports both Full and Block AttnRes modes.

    Args:
        residual_mode: "full_attnres" or "block_attnres"
    """
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim,
                 residual_mode="block_attnres", use_checkpoint=False):
        super(MixerAttnRes, self).__init__()

        self.residual_mode = residual_mode
        self.use_checkpoint = use_checkpoint
        self.N = N

        if residual_mode == "full_attnres":
            self.blocks = nn.ModuleList([
                MixerBlockFullAttnRes(enc_dim=enc_dim, n_head=n_head,
                                      n_token=n_token, enc_s_dim=enc_s_dim)
                for _ in range(N)
            ])
        elif residual_mode == "block_attnres":
            self.blocks = nn.ModuleList([
                MixerBlockAttnRes(enc_dim=enc_dim, n_head=n_head,
                                  n_token=n_token, enc_s_dim=enc_s_dim)
                for _ in range(N)
            ])
        else:
            raise ValueError(f"Unknown residual_mode: {residual_mode}")

    def forward(self, V, E, edges_long, pos_enc):
        """
        Args:
            V: [bs, N_nodes, enc_dim] — encoder output (embedding)
            E: [bs, ne, enc_dim] — edge embeddings
            edges_long: [bs, ne, 2] — edge indices
            pos_enc: [bs, N_nodes, enc_s_dim] — positional encoding
        Returns:
            V_all: [bs, N_block, N_nodes, enc_dim] — per-block outputs for decoder
        """
        if self.residual_mode == "block_attnres":
            return self._forward_block(V, E, edges_long, pos_enc)
        else:
            return self._forward_full(V, E, edges_long, pos_enc)

    def _forward_block(self, V, E, edges_long, pos_enc):
        """Block AttnRes: inter-block attention, intra-block standard residuals."""
        # b_0 = embedding
        block_reps = [V]  # completed block representations
        partial_block = None
        V_all = []

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Copy list to avoid in-place mutation issues with checkpoint
                block_reps, partial_block, E = grad_checkpoint(
                    self._block_forward_wrapper, block, list(block_reps), partial_block,
                    E, edges_long, pos_enc,
                    use_reentrant=False
                )
            else:
                block_reps, partial_block, E = block(
                    block_reps, partial_block, E, edges_long, pos_enc
                )

            # At block boundary: finalize this block, start new one
            block_reps = list(block_reps)
            block_reps.append(partial_block)
            V_all.append(partial_block)
            partial_block = None

        V_all = torch.stack(V_all, dim=1)  # [bs, N_block, N, enc_dim]
        return V_all

    @staticmethod
    def _block_forward_wrapper(block, block_reps, partial_block, E, edges_long, pos_enc):
        return block(block_reps, partial_block, E, edges_long, pos_enc)

    def _forward_full(self, V, E, edges_long, pos_enc):
        """Full AttnRes: every sub-layer is a source."""
        # sources[0] = embedding
        sources = [V]
        V_all = []

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Copy list to avoid in-place mutation issues with checkpoint
                sources, E = grad_checkpoint(
                    self._full_forward_wrapper, block, list(sources), E,
                    edges_long, pos_enc,
                    use_reentrant=False
                )
            else:
                sources, E = block(sources, E, edges_long, pos_enc)

            # The last source in this block's outputs is the FFN output
            V_all.append(sources[-1])

        V_all = torch.stack(V_all, dim=1)  # [bs, N_block, N, enc_dim]
        return V_all

    @staticmethod
    def _full_forward_wrapper(block, sources, E, edges_long, pos_enc):
        return block(sources, E, edges_long, pos_enc)


# ============================================================
# Model
# ============================================================

class Model(nn.Module):
    def __init__(self,
                 space_size=3,
                 pos_enc_dim=5,
                 cond_dim=32,
                 N_block=4,
                 in_dim=4,
                 out_dim=4,
                 enc_dim=128,
                 n_head=4,
                 n_token=128,
                 dt: float = 0.05,
                 stepper_scheme="euler",
                 use_checkpoint=False,
                 residual_mode="block_attnres",
                 ):
        super(Model, self).__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.out_dim = out_dim
        self.residual_mode = residual_mode

        self.pos_enc_dim = pos_enc_dim
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = (1 + 2 * pos_enc_dim) * 2
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.encoder = Encoder(
            space_size=space_size,
            state_size=in_dim,
            enc_dim=enc_dim,
            enc_t_dim=enc_t_dim,
            enc_c_dim=enc_c_dim
        )

        self.mixer = MixerAttnRes(
            N=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            residual_mode=residual_mode,
            use_checkpoint=use_checkpoint,
        )

        self.decoder = NSDecomposedDecoder(
            N=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            state_size=out_dim
        )

    @staticmethod
    def get_velocity_axis_info(field_names):
        axis_info = []
        for axis_id, field_name in enumerate(VELOCITY_FIELD_NAMES):
            if field_name in field_names:
                axis_info.append((axis_id, field_name, field_names.index(field_name)))
        return axis_info

    def forward(self, state_in, node_pos, edges, time_i, conditions, pos_enc=None, c_enc=None, dt=None):
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
            c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        if len(time_i.shape) == 1:
            time_i = time_i.view(-1, 1)
        bs = time_i.shape[0]

        if dt is None:
            dt_tensor = torch.full((bs, 1), self.dt, dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (float, int)):
            dt_tensor = torch.full((bs, 1), float(dt), dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (np.floating, np.integer)):
            dt_tensor = torch.tensor([dt], dtype=time_i.dtype, device=time_i.device).reshape(bs, 1)
        else:
            dt_tensor = dt.view(bs, 1).to(dtype=time_i.dtype, device=time_i.device)

        time_info = torch.cat([time_i, dt_tensor], dim=-1)

        t_enc = FourierEmbedding(time_info, 0, self.pos_enc_dim) # 时间编码
        edges_long = edges.long() if edges.dtype != torch.long else edges
        V, E = self.encoder(node_pos, state_in, t_enc, c_enc, edges_long)

        V_all = self.mixer(V, E, edges_long, pos_enc)

        delta_pred = self.decoder(V_all, pos_enc)

        state_pred = state_in + delta_pred * dt

        return state_pred

    def autoregressive(self,
                       state_in,
                       node_pos,
                       edges,
                       time_seq,
                       conditions,
                       dt=None,
                       teacher_forcing=False,
                       gt_states=None):

        state_t = state_in
        outputs = [state_in]
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        T = time_seq.shape[1]
        for t in range(T):
            time_i = time_seq[:, t]

            if self.training and t > 0:
                vel_mag = state_t.abs().clamp_min(1e-3)
                noise_scale = 0.005
                noise = torch.randn_like(state_t) * noise_scale * vel_mag
                state_t_input = state_t + noise
            else:
                state_t_input = state_t

            state_pred = self.forward(state_t_input, node_pos, edges, time_i, conditions, pos_enc, c_enc, dt)

            outputs.append(state_pred)

            if t < T - 1:
                if teacher_forcing and gt_states is not None:
                    state_t = gt_states[:, t]
                else:
                    state_t = state_pred

        outputs = torch.stack(outputs[1:], dim=1)

        return outputs
