"""
PhysGTO with Attention Residuals + Cross-Field Coupling.

Extends physgto_attnres.py with a lightweight CrossFieldAttention module
that enables explicit interaction between different physical field components
(e.g., Ux, Uy, Uz) within each MixerBlock.

Design rationale for velocity field prediction:
- Full separation (separate backbones per field) is wasteful because Ux/Uy/Uz
  share the same mesh geometry and are tightly coupled via Navier-Stokes.
- Instead, we keep a shared GNN+Attention backbone and add a lightweight
  CrossFieldAttention that allows component-specific processing with coupling.
- The cross-field attention splits the embedding into per-field views, lets
  each field attend to all others, and merges back with gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from torch_scatter import scatter_mean, scatter_softmax

# Import shared components from physgto_attnres
from src.physgto_attnres import (
    VELOCITY_FIELD_NAMES,
    get_edge_info,
    broadcast_dt,
    MLP,
    Atten,
    FourierEmbedding,
    GatedGNN,
    OperatorHead,
    NSDecomposedDecoder,
    Encoder,
    RMSNorm,
    attn_res_op,
)


# ============================================================
# Cross-Field Attention
# ============================================================

class CrossFieldAttention(nn.Module):
    """
    Lightweight cross-field attention for multi-physics coupling.

    Splits the node embedding into per-field views, applies cross-attention
    between field tokens, and merges back with a gated residual connection.

    This allows Ux to "see" Uy and Uz features (and vice versa) at each
    spatial location, enabling the model to learn inter-component coupling
    (e.g., ∂Ux/∂y affecting Uy through pressure, or continuity ∇·U=0).

    Args:
        enc_dim: embedding dimension (must be divisible by n_fields)
        n_fields: number of physical fields (3 for Ux,Uy,Uz)
        n_heads: number of attention heads for cross-field attention
    """
    def __init__(self, enc_dim, n_fields=3, n_heads=1):
        super().__init__()
        self.n_fields = n_fields
        assert enc_dim % n_fields == 0, f"enc_dim ({enc_dim}) must be divisible by n_fields ({n_fields})"
        self.field_dim = enc_dim // n_fields

        # Per-field projection: maps full embedding to field-decomposed space
        self.field_proj = nn.Linear(enc_dim, enc_dim)

        # Cross-field attention: each field token attends to all field tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.field_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # Merge back with gated residual
        self.merge = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(enc_dim)

    def forward(self, V):
        """
        Args:
            V: [B, N, enc_dim] — node embeddings
        Returns:
            V_out: [B, N, enc_dim] — cross-field enhanced embeddings
        """
        B, N, D = V.shape

        # Project and reshape to [B*N, n_fields, field_dim]
        h = self.field_proj(self.norm(V))
        h = h.reshape(B * N, self.n_fields, self.field_dim)

        # Cross-field attention: each field token attends to all field tokens
        h_cross, _ = self.cross_attn(h, h, h)

        # Reshape back to [B, N, enc_dim]
        h_cross = h_cross.reshape(B, N, D)

        # Gated residual connection
        gate = self.gate(V)
        out = V + gate * self.merge(h_cross)
        return out


# ============================================================
# Block AttnRes + CrossField: MixerBlock
# ============================================================

class MixerBlockCrossField(nn.Module):
    """
    MixerBlock with Block Attention Residuals + Cross-Field Attention.

    Flow: AttnRes → GNN → CrossFieldAttn → AttnRes → Atten → AttnRes → FFN
    """
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, n_fields=3):
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

        # Cross-field attention (inserted after GNN, before Perceiver attention)
        self.cross_field_attn = CrossFieldAttention(enc_dim=enc_dim, n_fields=n_fields)

        # Learnable residual scaling for intra-block
        self.alpha_gnn = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_cross = nn.Parameter(torch.ones(1) * 0.3)
        self.alpha_attn = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_ffn = nn.Parameter(torch.ones(1) * 0.5)

        # AttnRes pseudo-queries
        self.w_gnn = nn.Parameter(torch.zeros(enc_dim))
        self.w_attn = nn.Parameter(torch.zeros(enc_dim))
        self.w_ffn = nn.Parameter(torch.zeros(enc_dim))

        self.attn_res_norm = RMSNorm(enc_dim)

    def forward(self, blocks, partial_block, E, edges, s_enc):
        # --- AttnRes before GNN ---
        sources = blocks + ([partial_block] if partial_block is not None else [])
        h = attn_res_op(sources, self.w_gnn, self.attn_res_norm)

        # GNN sub-layer
        V_in = torch.cat([self.ln0(h), s_enc], dim=-1)
        v_gnn, e = self.gnn(V_in, E, edges)
        E = E + e
        if partial_block is not None:
            partial_block = partial_block + self.alpha_gnn * v_gnn
        else:
            partial_block = v_gnn

        # Cross-field attention (within intra-block residual)
        v_cross = self.cross_field_attn(partial_block)
        partial_block = partial_block + self.alpha_cross * (v_cross - partial_block)

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
# Full AttnRes + CrossField: MixerBlock
# ============================================================

class MixerBlockFullCrossField(nn.Module):
    """
    MixerBlock with Full Attention Residuals + Cross-Field Attention.
    """
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, n_fields=3):
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

        # Cross-field attention
        self.cross_field_attn = CrossFieldAttention(enc_dim=enc_dim, n_fields=n_fields)
        self.alpha_cross = nn.Parameter(torch.ones(1) * 0.3)

        # AttnRes pseudo-queries
        self.w_gnn = nn.Parameter(torch.zeros(enc_dim))
        self.w_attn = nn.Parameter(torch.zeros(enc_dim))
        self.w_ffn = nn.Parameter(torch.zeros(enc_dim))

        self.attn_res_norm = RMSNorm(enc_dim)

    def forward(self, sources, E, edges, s_enc):
        # Build a new list to avoid in-place mutation (required for checkpoint)
        new_sources = list(sources)

        # --- AttnRes before GNN ---
        h = attn_res_op(new_sources, self.w_gnn, self.attn_res_norm)

        # GNN sub-layer
        V_in = torch.cat([self.ln0(h), s_enc], dim=-1)
        v_gnn, e = self.gnn(V_in, E, edges)
        E = E + e
        new_sources.append(v_gnn)

        # Cross-field attention on GNN output
        v_cross = self.cross_field_attn(v_gnn)
        v_gnn_enhanced = v_gnn + self.alpha_cross * (v_cross - v_gnn)

        # --- AttnRes before Atten (using enhanced GNN output) ---
        # Replace last source with cross-field enhanced version
        new_sources[-1] = v_gnn_enhanced
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
# Mixer with CrossField + AttnRes
# ============================================================

class MixerCrossField(nn.Module):
    """
    Mixer with Cross-Field Attention and AttnRes.
    """
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim,
                 residual_mode="block_attnres", n_fields=3, use_checkpoint=False):
        super(MixerCrossField, self).__init__()

        self.residual_mode = residual_mode
        self.use_checkpoint = use_checkpoint
        self.N = N

        if residual_mode == "full_attnres":
            self.blocks = nn.ModuleList([
                MixerBlockFullCrossField(enc_dim=enc_dim, n_head=n_head,
                                         n_token=n_token, enc_s_dim=enc_s_dim,
                                         n_fields=n_fields)
                for _ in range(N)
            ])
        elif residual_mode == "block_attnres":
            self.blocks = nn.ModuleList([
                MixerBlockCrossField(enc_dim=enc_dim, n_head=n_head,
                                     n_token=n_token, enc_s_dim=enc_s_dim,
                                     n_fields=n_fields)
                for _ in range(N)
            ])
        else:
            raise ValueError(f"Unknown residual_mode: {residual_mode}")

    def forward(self, V, E, edges_long, pos_enc):
        if self.residual_mode == "block_attnres":
            return self._forward_block(V, E, edges_long, pos_enc)
        else:
            return self._forward_full(V, E, edges_long, pos_enc)

    def _forward_block(self, V, E, edges_long, pos_enc):
        block_reps = [V]
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

            block_reps = list(block_reps)
            block_reps.append(partial_block)
            V_all.append(partial_block)
            partial_block = None

        V_all = torch.stack(V_all, dim=1)
        return V_all

    @staticmethod
    def _block_forward_wrapper(block, block_reps, partial_block, E, edges_long, pos_enc):
        return block(block_reps, partial_block, E, edges_long, pos_enc)

    def _forward_full(self, V, E, edges_long, pos_enc):
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

            V_all.append(sources[-1])

        V_all = torch.stack(V_all, dim=1)
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
                 n_fields=3,
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

        self.mixer = MixerCrossField(
            N=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            residual_mode=residual_mode,
            n_fields=n_fields,
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
