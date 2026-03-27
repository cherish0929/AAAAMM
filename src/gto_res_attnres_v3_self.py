"""
PhysGTO-Res-AttnRes: Res 框架 + Block Attention Residuals + Multi-Field Cross-Attention

将 physgto_attnres_multi_v2 的全部功能同步到 physgto_res 框架:

physgto_res 特性:
  - [time_i, dt] 拼接 Fourier 编码
  - state_in + v_pred 直接增量预测 (不乘 dt)
  - teacher_forcing 支持

physgto_attnres_multi_v2 特性:
  - Block Attention Residuals (arXiv:2603.15031)
  - Multi-Field: 每场独立 GNN/Attn/FFN 分支
  - Cross-Field Attention: GatedCrossAttention / FieldCrossAttention
  - MultiFieldEncoder: 场间门控信息交换
  - MultiFieldDecoder: 每场独立 Decoder
  - attn_res_mode 开关: "block_inter" / "full"

n_fields 开关:
  - n_fields=1 (或不设): 单场模式, 退化为普通 physgto_res + AttnRes (无 cross-attention)
  - n_fields>=2: 多场模式, 完整的 Cross-Attention + Per-field 分支
"""

import torch
import torch.nn as nn
import numpy as np

from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint


# =============================================================================
# 工具函数
# =============================================================================

def get_edge_info(edges, node_pos):
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True))
    E = torch.cat([d, -d, norm], dim=-1)
    return E


def FourierEmbedding(pos, pos_start, pos_length):
    original_shape = pos.shape
    new_pos = pos.reshape(-1, original_shape[-1])
    index = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq = 2 ** index * torch.pi
    cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    embedding = torch.cat([cos_feat, sin_feat], dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, pos], dim=-1)

def FourierLinearEmbedding(info, max_freq, length):
    original_shape = info.shape
    new_info = info.reshape(-1, original_shape[-1])
    freq = torch.linspace(1, max_freq, length)
    cos_feat = torch.cos(freq.view(1, 1, -1) * new_info.unsqueeze(-1))
    sin_feat = torch.sin(freq.view(1, 1, -1) * new_info.unsqueeze(-1))
    embedding = torch.cat([cos_feat, sin_feat], dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, info], dim=-1)


# =============================================================================
# AttnRes 核心组件
# =============================================================================

class RMSNorm(nn.Module):
    """Parameter-free RMSNorm (论文 Eq.2)"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


def block_attn_res(blocks, partial_block, w, rms_norm):
    """
    Block Attention Residuals (论文 Section 3.2):
        V = [b_0, ..., b_{n-1}, partial_block]
        K = RMSNorm(V)
        alpha = softmax(w^T K)
        h = sum(alpha * V)
    """
    sources = blocks + [partial_block]
    V = torch.stack(sources, dim=0)                    # [S, bs, N, D]
    K = rms_norm(V)
    logits = torch.einsum('d, s b n d -> s b n', w, K)
    alpha = logits.softmax(dim=0)
    h = torch.einsum('s b n, s b n d -> b n d', alpha, V)
    return h


# =============================================================================
# 基础模块
# =============================================================================

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
            for i in range(1, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class Atten(nn.Module):
    """Projection-Inspired Attention: 三步 Q->W0, W->W, W0->W"""
    def __init__(self, n_token=128, c_dim=128, n_heads=4):
        super().__init__()
        self.c_dim = c_dim
        self.n_token = n_token
        self.n_heads = n_heads

        self.Q = nn.Parameter(torch.randn(self.n_token, self.c_dim), requires_grad=True)
        self.attention1 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)

    def forward(self, W0):
        batch = W0.shape[0]
        learned_Q = self.Q.unsqueeze(0).repeat(batch, 1, 1)
        W, _ = self.attention1(learned_Q, W0, W0)
        W, _ = self.attention2(W, W, W)
        W, _ = self.attention3(W0, W, W)
        return W


class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super().__init__()
        self.node_size = node_size
        self.output_size = output_size
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

    def get_edges_info(self, V, E, edges):
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        return edge_inpt

    def forward(self, V, E, edges):
        bs, N, _ = V.shape
        edge_inpt = self.get_edges_info(V, E, edges)
        edge_embeddings = self.f_edge(edge_inpt)

        edge_embeddings_0, edge_embeddings_1 = edge_embeddings.chunk(2, dim=-1)
        feat0 = edge_embeddings_0.shape[-1]
        feat1 = edge_embeddings_1.shape[-1]

        col_0 = edges[..., 0].unsqueeze(-1).expand(-1, -1, feat0)
        col_1 = edges[..., 1].unsqueeze(-1).expand(-1, -1, feat1)

        edge_mean_0 = scatter_mean(edge_embeddings_0, col_0, dim=1, dim_size=N)
        edge_mean_1 = scatter_mean(edge_embeddings_1, col_1, dim=1, dim_size=N)

        edge_mean = torch.cat([edge_mean_0, edge_mean_1], dim=-1)
        node_inpt = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


# =============================================================================
# Cross-Field Attention (来自 physgto_attnres_multi_v2)
# =============================================================================

class GatedCrossAttention(nn.Module):
    """
    门控 Projection-Inspired Cross-Field Attention (线性复杂度)
    3步 Projection-Inspired + 门控, 复杂度 O(N x n_token)
    """
    def __init__(self, enc_dim, n_heads=4, n_token=64):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(n_token, enc_dim))
        self.ln_other = nn.LayerNorm(enc_dim)
        self.ln_self = nn.LayerNorm(enc_dim)
        self.attn1 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        self.attn3 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, V_self, V_other):
        bs = V_self.shape[0]
        Q = self.Q.unsqueeze(0).expand(bs, -1, -1)
        other = self.ln_other(V_other)
        self_normed = self.ln_self(V_self)
        W, _ = self.attn1(Q, other, other)
        W, _ = self.attn2(W, W, W)
        out, _ = self.attn3(self_normed, W, W)
        return torch.tanh(self.gate) * out


class FieldCrossAttention(nn.Module):
    """
    Projection-Inspired Cross-Field Attention (用于 full 模式, 线性复杂度)
    3步: Q_tokens->V_other, tokens self-refine, V_self->tokens
    """
    def __init__(self, enc_dim, n_heads=4, n_token=64):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(n_token, enc_dim))
        self.ln_other = nn.LayerNorm(enc_dim)
        self.ln_self = nn.LayerNorm(enc_dim)
        self.attn1 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        self.attn3 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)

    def forward(self, V_self, V_other):
        bs = V_self.shape[0]
        Q = self.Q.unsqueeze(0).expand(bs, -1, -1)
        other = self.ln_other(V_other)
        self_normed = self.ln_self(V_self)
        W, _ = self.attn1(Q, other, other)
        W, _ = self.attn2(W, W, W)
        out, _ = self.attn3(self_normed, W, W)
        return out


# =============================================================================
# Encoder: 单场 / 多场自适应
# =============================================================================

class SingleFieldEncoder(nn.Module):
    """单场编码器 (与 physgto_res.py 一致)"""
    def __init__(self, space_size=3, state_size=4, enc_dim=128,
                 enc_t_dim=11, enc_c_dim=12):
        super().__init__()
        self.fv1 = MLP(input_size=state_size + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        state_in = torch.cat((state_in, node_pos), dim=-1)
        time_enc = self.fv_time(time_i)
        cond_enc = self.fv_cond(conditions)
        V = self.fv1(state_in) + time_enc.unsqueeze(-2) + cond_enc.unsqueeze(-2)
        E = self.fe(get_edge_info(edges, node_pos))
        return [V], E     # 返回 list 以统一接口


class MultiFieldEncoder(nn.Module):
    """
    多场编码器 (来自 physgto_attnres_multi_v2):
    - 每场独立 state MLP
    - 共享 time / condition / edge 编码器
    - 场间门控信息交换
    """
    def __init__(self, space_size=3, n_fields=2, enc_dim=128,
                 enc_t_dim=11, enc_c_dim=12):
        super().__init__()
        self.n_fields = n_fields

        self.fv_fields = nn.ModuleList([
            MLP(input_size=1 + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
            for _ in range(n_fields)
        ])

        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fuse_para = MLP(input_size=enc_dim * 2, output_size=enc_dim * 2, act='SiLU', layer_norm=False)
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

        # LayerNorm 层
        self.field_norms = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])

        self.exchange_norms = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])

        # 场间信息交换 (门控)
        self.field_exchange = nn.ModuleList([
            nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.SiLU())
            for _ in range(n_fields)
        ])
        self.field_exchange_gate = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_fields)
        ])

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        time_enc = self.fv_time(time_i)
        cond_enc = self.fv_cond(conditions) # [B, 128]

        # FiLM 调制，时间参数决定当前阶段
        h = torch.cat([cond_enc, time_enc], dim=-1) # [B, 256]
        para = self.fuse_para(h) # [B, 128], 不强迫处于同一语义空间
        gamma, beta = para.chunk(2, dim=-1) # [B, 128] / [B, 128]

        V_list = []
        for i in range(self.n_fields):
            field_i = state_in[..., i:i+1]
            inp = torch.cat([field_i, node_pos], dim=-1)
            V_local = self.fv_fields[i](inp)
            V_i = gamma.unsqueeze(-2) * V_local + beta.unsqueeze(-2)
            V_i = self.field_norms[i](V_i)
            V_list.append(V_i)

        # 场间信息交换
        V_exchanged = []
        for i in range(self.n_fields):
            other_sum = sum(V_list[j] for j in range(self.n_fields) if j != i)
            exchange_info = self.field_exchange[i](other_sum)
            gate = torch.tanh(self.field_exchange_gate[i])
            V_new = V_list[i] + gate * exchange_info
            V_new = self.exchange_norms[i](V_new)
            V_exchanged.append(V_new)

        E = self.fe(get_edge_info(edges, node_pos))
        return V_exchanged, E


# =============================================================================
# Decoder: 单场 / 多场自适应
# =============================================================================

class Decoder(nn.Module):
    """单场 Decoder (与 physgto_res.py 一致)"""
    def __init__(self, n_block=4, enc_dim=128, enc_s_dim=10, state_size=1):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(n_block * enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc):
        b, n_block, N, enc_dim = V_all.shape
        V_all = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        V = self.delta_net(torch.cat([V_all, pos_enc], dim=-1)) # 避免位置细节变模糊
        return V


class MultiFieldDecoder(nn.Module):
    """多场 Decoder: 每场独立 (来自 physgto_attnres_multi_v2)"""
    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10, n_fields=2):
        super().__init__()
        self.n_fields = n_fields
        self.decoders = nn.ModuleList([
            Decoder(n_block=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, state_size=1)
            for _ in range(n_fields)
        ])

    def forward(self, V_all_list, pos_enc):
        deltas = []
        for i in range(self.n_fields):
            delta_i = self.decoders[i](V_all_list[i], pos_enc)
            deltas.append(delta_i)
        return torch.cat(deltas, dim=-1)


# =============================================================================
# MixerBlock: 单场版 (n_fields=1 时使用)
# =============================================================================

class SingleFieldMixerBlock(nn.Module):
    """单场 MixerBlock + AttnRes (无 Cross-Attention)"""
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, attn_res_mode="block_inter"):
        super().__init__()
        self.attn_res_mode = attn_res_mode
        node_size = enc_dim + enc_s_dim

        self.gnn = GNN(node_size=node_size, edge_size=enc_dim, output_size=enc_dim, layer_norm=True)
        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.mha = Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head)
        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, 2 * enc_dim), nn.SiLU(), nn.Linear(2 * enc_dim, enc_dim)
        )

        if attn_res_mode == "full":
            self.attn_res_w = nn.ParameterList([nn.Parameter(torch.zeros(enc_dim)) for _ in range(3)])
        else:
            self.attn_res_w = nn.ParameterList([nn.Parameter(torch.zeros(enc_dim))])
        self.attn_res_norm = RMSNorm()

    def _forward_block_inter(self, V, E, edges, s_enc, blocks):
        norm = self.attn_res_norm
        h = block_attn_res(blocks, V, self.attn_res_w[0], norm)

        V_in = torch.cat([h, s_enc], dim=-1)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        h = h + v
        h = h + self.mha(self.ln1(h))
        h = h + self.ffn(self.ln2(h))

        blocks = blocks + [h]
        return h, E, blocks

    def _forward_full(self, V, E, edges, s_enc, blocks):
        norm = self.attn_res_norm

        h = block_attn_res(blocks, V, self.attn_res_w[0], norm)
        V_in = torch.cat([h, s_enc], dim=-1)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        partial = v

        h = block_attn_res(blocks, partial, self.attn_res_w[1], norm)
        partial = partial + self.mha(self.ln1(h))

        h = block_attn_res(blocks, partial, self.attn_res_w[2], norm)
        partial = partial + self.ffn(self.ln2(h))

        blocks = blocks + [partial]
        return partial, E, blocks

    def forward(self, V_list, E_list, edges, s_enc, blocks_list):
        """统一接口: V_list=[V], E_list=[E], blocks_list=[[...]]"""
        V, E, blocks = V_list[0], E_list[0], blocks_list[0]
        if self.attn_res_mode == "full":
            V, E, blocks = self._forward_full(V, E, edges, s_enc, blocks)
        else:
            V, E, blocks = self._forward_block_inter(V, E, edges, s_enc, blocks)
        return [V], [E], [blocks]


# =============================================================================
# MixerBlock: 多场版 (n_fields>=2, 来自 physgto_attnres_multi_v2)
# =============================================================================

class MultiFieldMixerBlock(nn.Module):
    """
    多场 MixerBlock + AttnRes + Cross-Field Attention

    attn_res_mode:
      "block_inter": Block 开头 1 次 AttnRes + 内部标准残差 + GatedCrossAttention
      "full":        每子层前 AttnRes + FieldCrossAttention
    """
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, n_fields=2,
                 cross_attn_heads=4, attn_res_mode="block_inter"):
        super().__init__()
        self.n_fields = n_fields
        self.attn_res_mode = attn_res_mode
        node_size = enc_dim + enc_s_dim

        # Per-field GNN
        self.gnns = nn.ModuleList([
            GNN(node_size=node_size, edge_size=enc_dim, output_size=enc_dim, layer_norm=True)
            for _ in range(n_fields)
        ])

        # Cross-Field Attention
        if attn_res_mode == "full":
            self.cross_attns = nn.ModuleList([
                FieldCrossAttention(enc_dim, n_heads=cross_attn_heads, n_token=n_token)
                for _ in range(n_fields)
            ])
        else:
            self.cross_attns = nn.ModuleList([
                GatedCrossAttention(enc_dim, n_heads=cross_attn_heads, n_token=n_token)
                for _ in range(n_fields)
            ])

        # Per-field Attention
        self.ln1s = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])
        self.mhas = nn.ModuleList([
            Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head)
            for _ in range(n_fields)
        ])

        # Per-field FFN
        self.ln2s = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(enc_dim, 2 * enc_dim), nn.SiLU(), nn.Linear(2 * enc_dim, enc_dim))
            for _ in range(n_fields)
        ])

        # AttnRes pseudo-query (零初始化)
        if attn_res_mode == "full":
            self.attn_res_w = nn.ParameterList([
                nn.Parameter(torch.zeros(enc_dim)) for _ in range(n_fields * 3)
            ])
        else:
            self.attn_res_w = nn.ParameterList([
                nn.Parameter(torch.zeros(enc_dim)) for _ in range(n_fields)
            ])
        self.attn_res_norm = RMSNorm()

    def _get_w(self, field_idx, sublayer_idx):
        return self.attn_res_w[field_idx * 3 + sublayer_idx]

    def _forward_block_inter(self, V_list, E_list, edges, s_enc, blocks_list):
        norm = self.attn_res_norm
        V_out, E_out = [], []

        # Step 1: Per-field AttnRes + GNN
        for i in range(self.n_fields):
            blocks_i, V_i, E_i = blocks_list[i], V_list[i], E_list[i]
            w = self.attn_res_w[i]
            h = block_attn_res(blocks_i, V_i, w, norm)

            V_in = torch.cat([h, s_enc], dim=-1)
            v, e = self.gnns[i](V_in, E_i, edges)
            E_i = E_i + e
            h = h + v
            V_out.append(h)
            E_out.append(E_i)

        # Step 2: Cross-Field Attention (门控残差)
        V_cross = []
        for i in range(self.n_fields):
            other_fields = [V_out[j] for j in range(self.n_fields) if j != i]
            if len(other_fields) == 1:
                cross_info = self.cross_attns[i](V_out[i], other_fields[0])
            else:
                other_cat = torch.cat(other_fields, dim=-2)
                cross_info = self.cross_attns[i](V_out[i], other_cat)
            V_cross.append(V_out[i] + cross_info)

        # Step 3: Per-field Attention (标准残差)
        V_attn = []
        for i in range(self.n_fields):
            h = V_cross[i]
            h = h + self.mhas[i](self.ln1s[i](h))
            V_attn.append(h)

        # Step 4: Per-field FFN (标准残差)
        V_final = []
        for i in range(self.n_fields):
            h = V_attn[i]
            h = h + self.ffns[i](self.ln2s[i](h))
            V_final.append(h)

        for i in range(self.n_fields):
            blocks_list[i] = blocks_list[i] + [V_final[i]]

        return V_final, E_out, blocks_list

    def _forward_full(self, V_list, E_list, edges, s_enc, blocks_list):
        norm = self.attn_res_norm
        V_out, E_out = [], []

        # Step 1: Per-field GNN with AttnRes
        for i in range(self.n_fields):
            blocks_i, V_i, E_i = blocks_list[i], V_list[i], E_list[i]
            w_gnn = self._get_w(i, 0)
            h = block_attn_res(blocks_i, V_i, w_gnn, norm)

            V_in = torch.cat([h, s_enc], dim=-1)
            v, e = self.gnns[i](V_in, E_i, edges)
            E_i = E_i + e
            V_out.append(v)
            E_out.append(E_i)

        # Step 2: Cross-Field Attention
        V_cross = []
        for i in range(self.n_fields):
            other_fields = [V_out[j] for j in range(self.n_fields) if j != i]
            if len(other_fields) == 1:
                cross_info = self.cross_attns[i](V_out[i], other_fields[0])
            else:
                other_cat = torch.cat(other_fields, dim=-2)
                cross_info = self.cross_attns[i](V_out[i], other_cat)
            V_cross.append(V_out[i] + cross_info)

        # Step 3: Per-field Attention with AttnRes
        V_attn = []
        for i in range(self.n_fields):
            partial = V_cross[i]
            w_attn = self._get_w(i, 1)
            h = block_attn_res(blocks_list[i], partial, w_attn, norm)
            partial = partial + self.mhas[i](self.ln1s[i](h))
            V_attn.append(partial)

        # Step 4: Per-field FFN with AttnRes
        V_final = []
        for i in range(self.n_fields):
            partial = V_attn[i]
            w_ffn = self._get_w(i, 2)
            h = block_attn_res(blocks_list[i], partial, w_ffn, norm)
            partial = partial + self.ffns[i](self.ln2s[i](h))
            V_final.append(partial)

        for i in range(self.n_fields):
            blocks_list[i] = blocks_list[i] + [V_final[i]]

        return V_final, E_out, blocks_list

    def forward(self, V_list, E_list, edges, s_enc, blocks_list):
        if self.attn_res_mode == "full":
            return self._forward_full(V_list, E_list, edges, s_enc, blocks_list)
        else:
            return self._forward_block_inter(V_list, E_list, edges, s_enc, blocks_list)


# =============================================================================
# Mixer: 统一入口
# =============================================================================

class Mixer(nn.Module):
    def __init__(self, N_block, enc_dim, n_head, n_token, enc_s_dim,
                 n_fields=1, cross_attn_heads=4, attn_res_mode="block_inter"):
        super().__init__()
        self.n_fields = n_fields

        if n_fields == 1:
            self.blocks = nn.ModuleList([
                SingleFieldMixerBlock(
                    enc_dim=enc_dim, n_head=n_head, n_token=n_token,
                    enc_s_dim=enc_s_dim, attn_res_mode=attn_res_mode
                )
                for _ in range(N_block)
            ])
        else:
            self.blocks = nn.ModuleList([
                MultiFieldMixerBlock(
                    enc_dim=enc_dim, n_head=n_head, n_token=n_token,
                    enc_s_dim=enc_s_dim, n_fields=n_fields,
                    cross_attn_heads=cross_attn_heads, attn_res_mode=attn_res_mode
                )
                for _ in range(N_block)
            ])

    def forward(self, V_list, E, edges, pos_enc):
        # 初始化 blocks_list: b_0 = encoder embedding
        blocks_list = [[V_list[i]] for i in range(self.n_fields)]

        # 各场独立演化边特征
        E_list = [E.clone() for _ in range(self.n_fields)]

        V_all = [[] for _ in range(self.n_fields)]

        for block in self.blocks:
            V_list, E_list, blocks_list = block(V_list, E_list, edges, pos_enc, blocks_list)
            for i in range(self.n_fields):
                V_all[i].append(V_list[i])

        # Stack: [bs, N_block, N, enc_dim] per field
        V_all_stacked = [torch.stack(V_all[i], dim=1) for i in range(self.n_fields)]
        return V_all_stacked


# =============================================================================
# 完整模型
# =============================================================================

class Model(nn.Module):
    """
    PhysGTO-Res-AttnRes

    physgto_res 框架 + AttnRes + Multi-Field Cross-Attention:
    - 时间编码: [time_i, dt] 拼接后 Fourier (来自 physgto_res)
    - 状态更新: state_in + v_pred 直接增量 (来自 physgto_res)
    - teacher_forcing 支持 (来自 physgto_res)
    - AttnRes: block_inter / full 可选 (来自 attnres 论文)
    - n_fields=1: 单场模式 (无 cross-attention)
    - n_fields>=2: 多场模式 (per-field 分支 + cross-attention)
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
                 n_token=128,
                 dt: float = 0.05,
                 stepper_scheme="euler",
                 n_fields=None,
                 cross_attn_heads=4,
                 attn_res_mode="block_inter",
                 ):
        super().__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.pos_enc_dim = pos_enc_dim
        self.n_fields = n_fields if n_fields is not None else in_dim
        self.attn_res_mode = attn_res_mode

        assert attn_res_mode in ("block_inter", "full"), \
            f"attn_res_mode must be 'block_inter' or 'full', got '{attn_res_mode}'"

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 2 * (1 + 2 * pos_enc_dim)    # [time_i, dt] 拼接
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        # Encoder
        if self.n_fields == 1:
            self.encoder = SingleFieldEncoder(
                space_size=space_size, state_size=in_dim, enc_dim=enc_dim,
                enc_t_dim=enc_t_dim, enc_c_dim=enc_c_dim,
            )
        else:
            self.encoder = MultiFieldEncoder(
                space_size=space_size, n_fields=self.n_fields, enc_dim=enc_dim,
                enc_t_dim=enc_t_dim, enc_c_dim=enc_c_dim,
            )

        # Mixer
        self.mixer = Mixer(
            N_block=N_block, enc_dim=enc_dim, n_head=n_head, n_token=n_token,
            enc_s_dim=enc_s_dim, n_fields=self.n_fields,
            cross_attn_heads=cross_attn_heads, attn_res_mode=attn_res_mode,
        )

        # Decoder
        if self.n_fields == 1:
            self.decoder = Decoder(n_block=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, state_size=out_dim)
        else:
            self.decoder = MultiFieldDecoder(
                N_block=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, n_fields=self.n_fields,
            )

    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):

        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
            c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        if len(time_i.shape) == 1:
            time_i = time_i.view(-1, 1)
        bs = time_i.shape[0]

        # dt 处理 (与 physgto_res 一致)
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

        # Encoder
        V_list, E = self.encoder(node_pos, state_in, t_enc, c_enc, edges_long)

        # Mixer
        V_all_list = self.mixer(V_list, E, edges_long, pos_enc)

        # Decoder
        if self.n_fields == 1:
            v_pred = self.decoder(V_all_list[0], pos_enc)
        else:
            v_pred = self.decoder(V_all_list, pos_enc)

        # 直接增量预测 (来自 physgto_res)
        state_pred = state_in + v_pred
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

        state_t = state_in
        outputs = [state_in]
        T = time_seq.shape[1]

        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        for t in range(T):
            time_i = time_seq[:, t]

            def custom_forward(s_t, t_i):
                return self.forward(s_t, node_pos, edges, t_i, conditions, pos_enc, c_enc, dt)

            if check_point:
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                state_pred = checkpoint(custom_forward, state_t, time_i, use_reentrant=False)
            else:
                state_pred = self.forward(state_t, node_pos, edges, time_i, conditions, pos_enc, c_enc, dt)

            outputs.append(state_pred)

            if t < T - 1:
                if teacher_forcing and gt_states is not None:
                    state_t = gt_states[:, t]
                else:
                    state_t = state_pred

        outputs = torch.stack(outputs[1:], dim=1)
        return outputs


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    print("=" * 60)
    print("PhysGTO-Res-AttnRes 快速验证")
    print("=" * 60)

    bs, N, ne = 2, 64, 128
    T = 4
    space_dim = 3
    cond_dim = 8

    # ---- 测试 1: 单场模式 (n_fields=1) ----
    for mode in ["block_inter", "full"]:
        print(f"\n{'--'*20}")
        print(f"  n_fields=1, mode={mode}")
        print(f"{'--'*20}")

        in_dim = out_dim = 4
        model = Model(
            space_size=space_dim, pos_enc_dim=3, cond_dim=cond_dim,
            N_block=4, in_dim=in_dim, out_dim=out_dim, enc_dim=64,
            n_head=4, n_token=32, dt=2e-5,
            n_fields=1, attn_res_mode=mode,
        )

        state_in = torch.randn(bs, N, in_dim)
        node_pos = torch.rand(bs, N, space_dim)
        edges = torch.randint(0, N, (bs, ne, 2))
        time_seq = torch.linspace(0, 1e-4, T).unsqueeze(0).expand(bs, -1)
        conditions = torch.randn(bs, cond_dim)

        pred = model(state_in, node_pos, edges, time_seq[:, 0], conditions)
        assert pred.shape == (bs, N, out_dim), f"Expected {(bs, N, out_dim)}, got {pred.shape}"
        print(f"[单步] {pred.shape} OK")

        out = model.autoregressive(state_in, node_pos, edges, time_seq, conditions)
        assert out.shape == (bs, T, N, out_dim)
        print(f"[自回归] {out.shape} OK")

        total = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total/1e6:.3f}M")

    # ---- 测试 2: 多场模式 (n_fields=2) ----
    for mode in ["block_inter", "full"]:
        print(f"\n{'--'*20}")
        print(f"  n_fields=2, mode={mode}")
        print(f"{'--'*20}")

        in_dim = out_dim = 2
        model = Model(
            space_size=space_dim, pos_enc_dim=3, cond_dim=cond_dim,
            N_block=4, in_dim=in_dim, out_dim=out_dim, enc_dim=64,
            n_head=4, n_token=32, dt=2e-5,
            n_fields=2, cross_attn_heads=4, attn_res_mode=mode,
        )

        state_in = torch.randn(bs, N, in_dim)
        node_pos = torch.rand(bs, N, space_dim)
        edges = torch.randint(0, N, (bs, ne, 2))
        time_seq = torch.linspace(0, 1e-4, T).unsqueeze(0).expand(bs, -1)
        conditions = torch.randn(bs, cond_dim)

        pred = model(state_in, node_pos, edges, time_seq[:, 0], conditions)
        assert pred.shape == (bs, N, out_dim), f"Expected {(bs, N, out_dim)}, got {pred.shape}"
        print(f"[单步] {pred.shape} OK")

        out = model.autoregressive(state_in, node_pos, edges, time_seq, conditions)
        assert out.shape == (bs, T, N, out_dim)
        print(f"[自回归] {out.shape} OK")

        # checkpoint backward
        model.train()
        out_ck = model.autoregressive(state_in, node_pos, edges, time_seq, conditions, check_point=True)
        out_ck.sum().backward()
        print(f"[checkpoint] backward OK")

        total = sum(p.numel() for p in model.parameters())
        attnres_p = sum(p.numel() for n, p in model.named_parameters() if 'attn_res' in n)
        gate_p = sum(p.numel() for n, p in model.named_parameters() if 'gate' in n)
        print(f"参数量: {total/1e6:.3f}M (AttnRes: {attnres_p}, Gates: {gate_p})")

        # 零初始化检查
        for name, p in model.named_parameters():
            if 'attn_res_w' in name:
                assert torch.all(p == 0), f"{name} not zero!"
        print("AttnRes pseudo-query 零初始化 OK")

        if mode == "block_inter":
            for name, p in model.named_parameters():
                if 'gate' in name:
                    assert torch.all(p == 0), f"{name} not zero!"
            print("Gate 零初始化 OK")

    print(f"\n{'='*60}")
    print("全部验证通过!")
    print(f"{'='*60}")
