import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from torch_scatter import scatter_mean, scatter_softmax

VELOCITY_FIELD_NAMES = ("Ux", "Uy", "Uz")

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

# ---------------------------
# Core modules
# ---------------------------
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


# ========================================================================
# Physics-Informed Decoder (PAPM-inspired)
#
# Adapts PAPM's core insight to graph-based architectures:
#   - Hard-coded differential operators (gradient, Laplacian) on the graph
#     with NO trainable parameters (like PAPM's fixed Conv2d kernels)
#   - Only scalar coefficients (viscosity, convection scale) are learned
#   - Pressure and source terms remain fully learned (no simple operator form)
#   - Physics-informed residual branch corrects operator approximation errors
#   - Spatially-adaptive combination merges physics + learned terms
#
# This replaces pure-MLP heads with structurally correct physics operators,
# providing a strong inductive bias matching the NS momentum equation:
#   ∂(ρU)/∂t = ∇·(μ∇U) - (U·∇)U - ∇p + F_s + F_m + F_damp
# ========================================================================

class GraphDiffOps(nn.Module):
    """
    Fixed (non-trainable) graph-based differential operators.
    Computes gradient, Laplacian, and convection on irregular 3D graphs
    using axis-aligned finite differences.

    Inspired by PAPM's hard-coded Conv2d Laplacian/gradient kernels,
    adapted for unstructured graph connectivity.
    """
    def __init__(self, space_dim=3, eps=1e-8):
        super().__init__()
        self.space_dim = space_dim
        self.eps = eps

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, state_in, node_pos, edges):
        """
        Args:
            state_in: [bs, N, C] velocity field (C=3 for Ux,Uy,Uz)
            node_pos: [bs, N, 3] node coordinates
            edges:    [bs, E, 2] undirected edge indices (long)
        Returns:
            grad_U:  [bs, N, C, 3] velocity gradient (Jacobian) tensor
            lap_U:   [bs, N, C]    Laplacian of velocity
            conv_U:  [bs, N, C]    convection term (U·∇)U
        """
        bs, N, C = state_in.shape
        E_num = edges.shape[1]
        space_dim = self.space_dim
        eps = self.eps

        send_idx = edges[..., 0]  # [bs, E]
        recv_idx = edges[..., 1]  # [bs, E]

        # Gather positions and velocities at edge endpoints
        send_pos = torch.gather(node_pos, 1, send_idx.unsqueeze(-1).expand(-1, -1, space_dim))
        recv_pos = torch.gather(node_pos, 1, recv_idx.unsqueeze(-1).expand(-1, -1, space_dim))
        delta_pos = recv_pos - send_pos  # [bs, E, 3]

        send_vel = torch.gather(state_in, 1, send_idx.unsqueeze(-1).expand(-1, -1, C))
        recv_vel = torch.gather(state_in, 1, recv_idx.unsqueeze(-1).expand(-1, -1, C))
        delta_vel = recv_vel - send_vel  # [bs, E, C]

        # ---- Gradient: axis-aligned finite differences ----
        # For each edge, determine the primary spatial axis
        abs_delta = delta_pos.abs()
        edge_axis = abs_delta.argmax(dim=-1)  # [bs, E], values in {0,1,2}

        # Spacing along primary axis (signed, for correct gradient direction)
        spacing = delta_pos.gather(-1, edge_axis.unsqueeze(-1)).squeeze(-1)  # [bs, E]
        spacing = spacing + eps * spacing.sign()  # avoid division by zero, preserve sign
        # Handle exactly-zero spacing (shouldn't happen on valid grids)
        spacing = torch.where(spacing.abs() < eps, torch.full_like(spacing, eps), spacing)

        # Per-edge gradient: dU_k / dx_a = delta_vel_k / spacing
        edge_grad = delta_vel / spacing.unsqueeze(-1)  # [bs, E, C]

        # Scatter gradients to nodes, accumulating per-axis
        # We need grad_U[bs, N, C, 3] — so we scatter into a [bs, N*C*3] buffer
        # using a combined index = node_idx * C * 3 + vel_comp * 3 + axis
        # But simpler: loop over 3 axes and scatter masked edges

        grad_sum = state_in.new_zeros(bs, N, C, space_dim)
        grad_count = state_in.new_zeros(bs, N, space_dim)

        for a in range(space_dim):
            axis_mask = (edge_axis == a).unsqueeze(-1)  # [bs, E, 1]
            masked_grad = edge_grad * axis_mask.float()  # [bs, E, C]
            count_val = axis_mask.float().squeeze(-1)    # [bs, E]

            # Scatter to sender nodes (node i gets gradient from edge i->j)
            send_exp = send_idx.unsqueeze(-1).expand(-1, -1, C)  # [bs, E, C]
            grad_sum[:, :, :, a].scatter_add_(1, send_exp, masked_grad)
            grad_count[:, :, a].scatter_add_(1, send_idx, count_val)

            # Scatter to receiver nodes (node j gets same gradient value)
            recv_exp = recv_idx.unsqueeze(-1).expand(-1, -1, C)
            grad_sum[:, :, :, a].scatter_add_(1, recv_exp, masked_grad)
            grad_count[:, :, a].scatter_add_(1, recv_idx, count_val)

        # Average by count
        grad_count = grad_count.clamp_min(1.0)
        grad_U = grad_sum / grad_count.unsqueeze(2)  # [bs, N, C, 3]

        # ---- Laplacian: weighted neighbor differences ----
        # lap_U_i = sum_j (U_j - U_i) / dist_ij^2, normalized by neighbor count
        dist_sq = (delta_pos ** 2).sum(-1).clamp_min(eps)  # [bs, E]
        weight = 1.0 / dist_sq  # [bs, E]

        weighted_diff = delta_vel * weight.unsqueeze(-1)  # [bs, E, C]

        lap_sum = state_in.new_zeros(bs, N, C)
        lap_count = state_in.new_zeros(bs, N)

        send_exp_c = send_idx.unsqueeze(-1).expand(-1, -1, C)
        recv_exp_c = recv_idx.unsqueeze(-1).expand(-1, -1, C)

        # For undirected edge (i,j): node i gets +(U_j - U_i)/d^2
        #                             node j gets -(U_j - U_i)/d^2 = +(U_i - U_j)/d^2
        lap_sum.scatter_add_(1, send_exp_c, weighted_diff)
        lap_sum.scatter_add_(1, recv_exp_c, -weighted_diff)

        ones_e = weight.new_ones(bs, E_num)
        lap_count.scatter_add_(1, send_idx, ones_e)
        lap_count.scatter_add_(1, recv_idx, ones_e)

        lap_U = lap_sum / lap_count.unsqueeze(-1).clamp_min(1.0)  # [bs, N, C]

        # ---- Convection: (U · ∇)U ----
        # conv_k = sum_j U_j * dU_k/dx_j
        conv_U = torch.einsum('bnj, bncj -> bnc', state_in, grad_U)  # [bs, N, C]

        return grad_U, lap_U, conv_U


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


class PhysicsInformedDecoder(nn.Module):
    """
    PAPM-inspired decoder with hard-coded graph differential operators.

    Combines fixed physics operators (gradient, Laplacian, convection) with
    learned branches (pressure, source, residual) via spatially-adaptive gating.

    Physics branches (diffusion, convection):
        - Differential operators are FIXED (non-trainable), computed by GraphDiffOps
        - Only scalar coefficients (viscosity ν, convection scale α) are learned
        - This prevents collapse into arbitrary MLPs

    Learned branches (pressure, source, residual):
        - Capture terms without simple operator form (∇p, F_s, F_m, F_damp)
        - Use encoder features from different mixer blocks

    The combiner receives both encoder features AND physics features,
    enabling context-dependent trust weighting.
    """
    def __init__(self, N=4, enc_dim=128, enc_s_dim=10, state_size=3, space_dim=3):
        super().__init__()
        self.state_size = state_size
        self.N = N
        self.enc_dim = enc_dim
        self.space_dim = space_dim
        in_dim = N * enc_dim + enc_s_dim

        # Physics feature dimension: lap(C) + conv(C) + grad_flat(C*3)
        phys_feat_dim = state_size + state_size + state_size * space_dim  # 3+3+9=15

        # Fixed graph differential operators (NO trainable parameters)
        self.diff_ops = GraphDiffOps(space_dim=space_dim)

        # Shared projection from concatenated block features
        self.proj = nn.Linear(in_dim, enc_dim)
        self.proj_norm = nn.LayerNorm(enc_dim)

        # ---- Diffusion branch: ν(x) * ∇²U ----
        # ν is learned from local encoder features (spatially varying viscosity)
        # Laplacian is computed by GraphDiffOps (fixed, non-trainable)
        self.visc_net = nn.Sequential(
            nn.Linear(enc_dim + enc_s_dim, enc_dim // 2),
            nn.SiLU(),
            nn.Linear(enc_dim // 2, 1),
            nn.Softplus(),  # viscosity must be positive
        )

        # ---- Convection branch: -α(x) * (U·∇)U ----
        # α is learned scale (absorbs density and normalization factors)
        # Convection operator is computed by GraphDiffOps (fixed)
        self.conv_scale_net = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.SiLU(),
            nn.Linear(enc_dim // 2, 1),
            nn.Sigmoid(),  # bounded [0, 1]
        )

        # ---- Pressure branch: global, long-range (∇p) ----
        # No simple fixed operator (pressure is implicitly coupled)
        # Uses features from FIRST mixer block (global attention patterns)
        self.press_proj = nn.Linear(enc_dim + enc_s_dim, enc_dim)
        self.press_res = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
        )
        self.press_head = OperatorHead(enc_dim, state_size)

        # ---- Source branch: localized forces (F_s, F_m, F_damp) ----
        # Deep processing with gating for complex force patterns
        self.src_res = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, enc_dim * 2),
            nn.SiLU(),
            nn.Linear(enc_dim * 2, enc_dim),
        )
        self.src_gate = nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.Sigmoid())
        self.src_head = OperatorHead(enc_dim, state_size)

        # ---- Residual correction branch ----
        # Receives both encoder features and physics features
        # Compensates for operator approximation errors on irregular grids
        self.resid_proj = nn.Sequential(
            nn.Linear(enc_dim + phys_feat_dim, enc_dim),
            nn.SiLU(),
        )
        self.resid_head = OperatorHead(enc_dim, state_size)

        # ---- Spatially-adaptive combination ----
        # 5 terms: diffusion, convection, pressure, source, residual
        # Combiner sees BOTH encoder and physics features for context-dependent trust
        self.combiner = nn.Sequential(
            nn.Linear(enc_dim + phys_feat_dim, enc_dim // 2),
            nn.SiLU(),
            nn.Linear(enc_dim // 2, 5),
        )

    def forward(self, V_all, pos_enc, state_in, node_pos, edges):
        """
        Args:
            V_all:    [bs, n_block, N, enc_dim] multi-scale encoder features
            pos_enc:  [bs, N, enc_s_dim] Fourier positional encoding
            state_in: [bs, N, state_size] current velocity field
            node_pos: [bs, N, 3] node coordinates
            edges:    [bs, E, 2] edge indices (long)
        Returns:
            delta:    [bs, N, state_size] predicted velocity change
        """
        b, n_block, N, enc_dim = V_all.shape

        # ===== Phase 1: Fixed physics operators (no trainable params) =====
        grad_U, lap_U, conv_U = self.diff_ops(state_in, node_pos, edges)
        # grad_U: [b, N, C, 3], lap_U: [b, N, C], conv_U: [b, N, C]

        # Flatten physics features for downstream use
        grad_flat = grad_U.reshape(b, N, -1)  # [b, N, C*3]
        phys_feat = torch.cat([lap_U, conv_U, grad_flat], dim=-1)  # [b, N, phys_feat_dim]

        # ===== Phase 2: Encoder feature processing =====
        V_first = V_all[:, 0]   # [b, N, enc_dim] — global/coarse
        V_last = V_all[:, -1]   # [b, N, enc_dim] — local/refined

        V_cat = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        h_full = self.proj_norm(self.proj(torch.cat([V_cat, pos_enc], dim=-1)))

        # ===== Phase 3: Physics-conditioned branches =====
        # Diffusion: ν(x) * lap_U
        nu = self.visc_net(torch.cat([V_last, pos_enc], dim=-1))  # [b, N, 1]
        delta_diff = nu * lap_U  # [b, N, C]

        # Convection: -α(x) * (U·∇)U
        alpha = self.conv_scale_net(h_full)  # [b, N, 1]
        delta_conv = -alpha * conv_U  # [b, N, C]

        # ===== Phase 4: Learned branches =====
        # Pressure: global, from first block's attention features
        h_press = self.press_proj(torch.cat([V_first, pos_enc], dim=-1))
        h_press = h_press + self.press_res(h_press)
        delta_press = self.press_head(h_press)  # [b, N, C]

        # Source: localized forces with gating
        h_src = h_full
        h_src = h_src + self.src_gate(h_src) * self.src_res(h_src)
        delta_src = self.src_head(h_src)  # [b, N, C]

        # Residual: corrects physics operator errors, sees both encoder + physics
        h_resid = self.resid_proj(torch.cat([h_full, phys_feat], dim=-1))
        delta_resid = self.resid_head(h_resid)  # [b, N, C]

        # ===== Phase 5: Spatially-adaptive combination =====
        combo_input = torch.cat([h_full, phys_feat], dim=-1)
        weights = F.softmax(self.combiner(combo_input), dim=-1)  # [b, N, 5]

        # Stack all deltas: [b, N, 5, C]
        deltas = torch.stack([delta_diff, delta_conv, delta_press, delta_src, delta_resid], dim=2)

        # Weighted sum: [b, N, C]
        delta = (weights.unsqueeze(-1) * deltas).sum(dim=2)

        return delta


class MixerBlock(nn.Module):
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

        self.alpha_gnn = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_attn = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_ffn = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, V, E, edges, s_enc):
        V_in = torch.cat([self.ln0(V), s_enc], dim=-1)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        V = V + self.alpha_gnn * v

        V = V + self.alpha_attn * self.mha(self.ln1(V))

        V = V + self.alpha_ffn * self.ffn(self.ln2(V))

        return V, E


class Encoder(nn.Module):
    """
    Enhanced encoder with FiLM conditioning.
    """
    def __init__(self,
                 space_size=2,
                 state_size=4,
                 enc_dim=128,
                 enc_t_dim = 11,
                 enc_c_dim = 12
                 ):
        super(Encoder, self).__init__()

        # +1 for velocity magnitude, +state_size for per-component sign indicators
        self.fv1 = MLP(input_size=state_size + space_size + 1 + state_size, output_size=enc_dim, act='SiLU', layer_norm=False)

        self.film_time = nn.Sequential(
            nn.Linear(enc_t_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim * 2),
        )
        self.film_cond = nn.Sequential(
            nn.Linear(enc_c_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim * 2),
        )

        # Edge: spatial (7) + vel diff (state_size) + mag diff (1)
        self.fe = MLP(input_size=2 * space_size + 1 + state_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        vel_mag = torch.norm(state_in, dim=-1, keepdim=True)
        vel_sign = torch.sign(state_in)

        state_aug = torch.cat((state_in, node_pos, vel_mag, vel_sign), dim=-1)
        V = self.fv1(state_aug)

        # FiLM conditioning (gamma clipped to prevent instability)
        time_film = self.film_time(time_i)
        t_gamma, t_beta = time_film.chunk(2, dim=-1)
        t_gamma = torch.tanh(t_gamma) * 0.5
        V = V * (1.0 + t_gamma.unsqueeze(-2)) + t_beta.unsqueeze(-2)

        cond_film = self.film_cond(conditions)
        c_gamma, c_beta = cond_film.chunk(2, dim=-1)
        c_gamma = torch.tanh(c_gamma) * 0.5
        V = V * (1.0 + c_gamma.unsqueeze(-2)) + c_beta.unsqueeze(-2)

        # Edge embedding
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

class Mixer(nn.Module):
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim, use_checkpoint=False):
        super(Mixer, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            MixerBlock(enc_dim=enc_dim, n_head=n_head, n_token=n_token, enc_s_dim=enc_s_dim)
            for _ in range(N)
        ])

    def forward(self, V, E, edges_long, pos_enc):

        V_all = []

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                V, E = grad_checkpoint(block, V, E, edges_long, pos_enc, use_reentrant=False)
            else:
                V, E = block(V, E, edges_long, pos_enc)
            V_all.append(V)

        V_all = torch.stack(V_all, dim=1) # [bs, N_block, N, enc_dim]

        return V_all

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
                 dt:float =0.05,
                 stepper_scheme="euler",
                 use_checkpoint=False,
                 ):
        super(Model, self).__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.out_dim = out_dim

        self.pos_enc_dim = pos_enc_dim
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = (1 + 2 * pos_enc_dim) * 2
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.encoder = Encoder(
            space_size = space_size,
            state_size = in_dim,
            enc_dim = enc_dim,
            enc_t_dim = enc_t_dim,
            enc_c_dim = enc_c_dim
            )

        self.mixer = Mixer(
            N=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            use_checkpoint=use_checkpoint,
            )

        # PAPM-inspired physics-informed decoder
        self.decoder = PhysicsInformedDecoder(
            N=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            state_size=out_dim,
            space_dim=space_size
            )

    @staticmethod
    def get_velocity_axis_info(field_names):
        axis_info = []
        for axis_id, field_name in enumerate(VELOCITY_FIELD_NAMES):
            if field_name in field_names:
                axis_info.append((axis_id, field_name, field_names.index(field_name)))
        return axis_info

    def forward(self, state_in, node_pos, edges, time_i, conditions, pos_enc = None, c_enc = None, dt=None):
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

        t_enc = FourierEmbedding(time_info, 0, self.pos_enc_dim)

        edges_long = edges.long() if edges.dtype != torch.long else edges

        V, E = self.encoder(node_pos, state_in, t_enc, c_enc, edges_long)

        V_all = self.mixer(V, E, edges_long, pos_enc)

        # Pass state_in, node_pos, edges to decoder for physics operators
        delta_pred = self.decoder(V_all, pos_enc, state_in, node_pos, edges_long)

        state_pred = state_in + delta_pred * broadcast_dt(dt_tensor, state_in)

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

            # Adaptive pushforward noise
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
