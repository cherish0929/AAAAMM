"""
PhysGTO-AttnRes-Multi-V3: Block Attention Residuals + Multi-Field Cross-Attention + V2 Improvements

基于 physgto_attnres_multi.py 的全面改进，同步了 physgto_v2.py 的所有更新：

1. Block Attention Residuals (arXiv:2603.15031)
   - 将标准残差连接 h_l = h_{l-1} + f_l(h_{l-1}) 替换为
     h_l = Σ α_{i→l} · v_i，其中 α 是通过 softmax attention 在深度维度上学习的权重
   - 每个子层（GNN / Attention / FFN）前都插入 block_attn_res
   - pseudo-query 初始化为零 → 初始时等价于标准残差

2. Multi-Field Cross-Attention
   - 不同物理场（如 T, alpha.air）使用独立的 GNN / Attention / FFN 分支
   - 通过 Cross-Attention 进行场间耦合
   - 共享边编码器，但各场独立演化边特征
   - 各场独立 Decoder 输出头

3. V2 Improvements (同步自 physgto_v2.py):
   - 各向异性位置编码 FourierEmbedding_pos (x 轴高频 boost)
   - GNN 注意力加权消息聚合 (scatter_softmax + scatter_add 取代 scatter_mean)
   - Decoder 注意力加权 block 聚合 (softmax over blocks 取代 concat)
   - Encoder FiLM 条件化 (乘法调制 γ·V + β 取代加法)
   - Encoder 新增 spatial_inform 输入（坐标范围、网格数量等空间信息）
   - 增强 Atten：Xavier 初始化、数据相关 query offset、多层 latent attention
   - 时间编码增强：[t,dt] Fourier + 单独 t 低频 Fourier
   - get_edge_info 数值稳定：+1e-8 防 NaN
   - logit clamping 防 softmax 溢出
   - 支持 stepper_scheme: "euler" / "delta"
   - checkpoint 模式下更安全的 detach().requires_grad_()
"""

import torch
import torch.nn as nn
import numpy as np

from torch.amp import GradScaler, autocast
from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from torch.utils.checkpoint import checkpoint


# =============================================================================
# 工具函数
# =============================================================================

def get_edge_info(edges, node_pos):
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True) + 1e-8)
    E = torch.cat([d, -d, norm], dim=-1)
    return E


def FourierEmbedding(pos, pos_start, pos_length):
    # F(x) = [cos(2^i * pi * x), sin(2^i * pi * x)]
    # 高频展开，目的是拟合复杂函数而非保留原有的维度
    original_shape = pos.shape
    new_pos = pos.reshape(-1, original_shape[-1])
    index = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq = 2 ** index * torch.pi
    cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    embedding = torch.cat([cos_feat, sin_feat], dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, pos], dim=-1)


def FourierEmbedding_pos(pos, pos_enc_dim, x_boost=2):
    """
    各向异性位置编码：x轴使用更高频（频率指数偏移 x_boost），y/z轴保持标准频率。
    pos: (..., space_size)，space_size >= 1，第0维为x，其余为y/z
    返回: (..., space_size + 2 * pos_enc_dim * space_size)
    """
    original_shape = pos.shape
    space_size = original_shape[-1]
    new_pos = pos.reshape(-1, space_size)  # (B*N, space_size)

    index = torch.arange(0, pos_enc_dim, device=pos.device).float()  # (pos_enc_dim,)

    parts = []
    for dim_i in range(space_size):
        start = x_boost if dim_i == 0 else 0
        freq = 2 ** (index + start) * torch.pi  # (pos_enc_dim,)
        xi = new_pos[:, dim_i:dim_i+1]           # (B*N, 1)
        cos_i = torch.cos(freq.unsqueeze(0) * xi)  # (B*N, pos_enc_dim)
        sin_i = torch.sin(freq.unsqueeze(0) * xi)  # (B*N, pos_enc_dim)
        parts.append(cos_i)
        parts.append(sin_i)

    embedding = torch.cat(parts, dim=-1)  # (B*N, 2 * pos_enc_dim * space_size)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, pos], dim=-1)


# =============================================================================
# 基础模块
# =============================================================================

class RMSNorm(nn.Module):
    """Parameter-free RMSNorm (用于 AttnRes 的 key 归一化)"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


def block_attn_res(blocks, partial_block, w, rms_norm):
    """
    Block Attention Residuals: 在深度维度上用 softmax attention 聚合历史块输出

    Args:
        blocks: list of [bs, N, D] tensors — 已完成的 block 表示 [b_0, ..., b_{n-1}]
        partial_block: [bs, N, D] — 当前 block 内的部分累加 b_n^i
        w: nn.Parameter of shape [D] — 伪查询向量 (初始化为零)
        rms_norm: RMSNorm instance — 对 key 做归一化

    Returns:
        h: [bs, N, D] — 加权聚合的隐状态
    """
    sources = blocks + [partial_block]
    V = torch.stack(sources, dim=0)               # [n+1, bs, N, D]
    K = rms_norm(V)                                # RMSNorm on keys
    logits = torch.einsum('d, s b n d -> s b n', w, K)  # [n+1, bs, N]
    logits = logits.clamp(-30, 30)
    alpha = logits.softmax(dim=0)                  # softmax over source dim
    h = torch.einsum('s b n, s b n d -> b n d', alpha, V)
    return h


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
    """
    增强版 Attention: Xavier 初始化 + 数据相关 query offset + 多层 latent attention
    (同步自 physgto_v2.py)
    """
    def __init__(self, n_token=128, c_dim=128, n_heads=4, n_latent=4):
        super().__init__()
        self.c_dim = c_dim
        self.n_token = n_token
        self.n_heads = n_heads
        self.n_latent = n_latent

        # Xavier 初始化，避免 randn 导致的注意力尺度问题
        self.Q = nn.Parameter(torch.empty(self.n_token, self.c_dim))
        nn.init.xavier_uniform_(self.Q)

        # W0 相关的 query offset
        self.q_offset = nn.Sequential(
            nn.Linear(self.c_dim, self.c_dim),
            nn.SiLU(),
            nn.Linear(self.c_dim, self.c_dim),
        )

        # Multihead attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, dropout=0.1, batch_first=True)
        self.attention2s = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, dropout=0.1, batch_first=True)
            for _ in range(self.n_latent)
        ])
        self.attention3 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)

    def forward(self, W0):
        # Step 1: Initial attention with learned query + W0-dependent offset
        batch = W0.shape[0]
        learned_Q = self.Q.unsqueeze(0).expand(batch, -1, -1)
        q_bias = self.q_offset(W0.mean(dim=1, keepdim=True))  # (batch, 1, c_dim)
        learned_Q = learned_Q + q_bias
        W, _ = self.attention1(learned_Q, W0, W0)

        # Step 2: Multi-layer self-attention on the transformed result
        for latent_atten in self.attention2s:
            W, _ = latent_atten(W, W, W)

        # Step 3: Position-aware attention
        W, _ = self.attention3(W0, W, W)
        return W


class GNN(nn.Module):
    """
    注意力加权消息聚合 GNN (同步自 physgto_v2.py)
    用 scatter_softmax + scatter_add 取代 scatter_mean
    """
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

        # 显式双头消息
        self.f_msg_sender = MLP(input_size=edge_size, output_size=edge_size // 2, n_hidden=n_hidden, act="SiLU", layer_norm=layer_norm)
        self.f_msg_receiver = MLP(input_size=edge_size, output_size=edge_size // 2, n_hidden=n_hidden, act="SiLU", layer_norm=layer_norm)

        # 注意力打分网络
        self.f_attn_sender = nn.Sequential(
            nn.Linear(edge_size, edge_size // 4),
            nn.SiLU(),
            nn.Linear(edge_size // 4, 1)
        )
        self.f_attn_receiver = nn.Sequential(
            nn.Linear(edge_size, edge_size // 4),
            nn.SiLU(),
            nn.Linear(edge_size // 4, 1)
        )

        self.f_node = MLP(
            input_size=edge_size + node_size,
            n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU',
            output_size=output_size
        )

    def forward(self, V, E, edges):
        bs, N, _ = V.shape
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        # 显式双头消息
        msg_sender = self.f_msg_sender(edge_embeddings)
        msg_receiver = self.f_msg_receiver(edge_embeddings)

        # 注意力加权聚合
        logit_sender = self.f_attn_sender(edge_embeddings).squeeze(-1)     # (bs, ne)
        logit_receiver = self.f_attn_receiver(edge_embeddings).squeeze(-1) # (bs, ne)

        # Clamp logits 防止 softmax 溢出
        logit_sender = logit_sender.clamp(-30, 30)
        logit_receiver = logit_receiver.clamp(-30, 30)

        feat0, feat1 = msg_sender.shape[-1], msg_receiver.shape[-1]

        col_0 = edges[..., 0].unsqueeze(-1).expand(-1, -1, feat0)
        col_1 = edges[..., 1].unsqueeze(-1).expand(-1, -1, feat1)
        col_0_scalar = edges[..., 0]  # (bs, ne)
        col_1_scalar = edges[..., 1]  # (bs, ne)

        # Per-node softmax attention weights
        alpha_0 = scatter_softmax(logit_sender, col_0_scalar, dim=1)    # (bs, ne)
        alpha_1 = scatter_softmax(logit_receiver, col_1_scalar, dim=1)  # (bs, ne)

        # Weighted aggregation
        edge_agg_0 = scatter_add(alpha_0.unsqueeze(-1) * msg_sender, col_0, dim=1, dim_size=N)
        edge_agg_1 = scatter_add(alpha_1.unsqueeze(-1) * msg_receiver, col_1, dim=1, dim_size=N)

        edge_agg = torch.cat([edge_agg_0, edge_agg_1], dim=-1)
        node_inpt = torch.cat([V, edge_agg], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


class FieldCrossAttention(nn.Module):
    """
    Projection-Inspired Cross-Field Attention (线性复杂度)

    避免 O(N²) 的全节点 cross-attention，改用学习的 query token 作为中介：
      1. Q_tokens attend to V_other → 压缩其他场信息到 n_token 个 token
      2. tokens self-refine
      3. V_self attend to refined tokens → 将跨场信息广播回每个节点

    复杂度: O(N × n_token)，而非 O(N²)
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
        """
        V_self:  [bs, N, enc_dim]
        V_other: [bs, N, enc_dim]
        Returns: [bs, N, enc_dim]
        """
        bs = V_self.shape[0]
        Q = self.Q.unsqueeze(0).expand(bs, -1, -1)
        other = self.ln_other(V_other)
        self_normed = self.ln_self(V_self)

        W, _ = self.attn1(Q, other, other)
        W, _ = self.attn2(W, W, W)
        out, _ = self.attn3(self_normed, W, W)
        return out


# =============================================================================
# Decoder (per-field, 注意力加权 block 聚合)
# =============================================================================

class Decoder(nn.Module):
    """
    注意力加权 block 聚合 Decoder (同步自 physgto_v2.py)
    用 softmax over blocks 取代 concat all blocks
    """
    def __init__(self, n_block=4, enc_dim=128, enc_s_dim=10, state_size=1):
        super().__init__()

        # 用 pos_enc 为每个节点在各 block 上生成注意力权重
        self.attn_net = nn.Sequential(
            nn.Linear(enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, n_block)
        )

        self.delta_net = nn.Sequential(
            nn.Linear(enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc):
        # V_all: [bs, n_block, N, enc_dim]
        # pos_enc: [bs, N, enc_s_dim]
        b, n_block, n_node, enc_dim = V_all.shape

        # attn_w: [bs, N, n_block] -> softmax over blocks
        attn_w = self.attn_net(pos_enc)                   # [bs, N, n_block]
        attn_w = attn_w.clamp(-30, 30)
        attn_w = torch.softmax(attn_w, dim=-1)            # [bs, N, n_block]

        # weighted sum over blocks
        V_perm = V_all.permute(0, 2, 1, 3)               # [bs, N, n_block, enc_dim]
        V_agg = (attn_w.unsqueeze(-1) * V_perm).sum(dim=2)  # [bs, N, enc_dim]

        V = self.delta_net(torch.cat([V_agg, pos_enc], dim=-1))
        return V


# =============================================================================
# Encoder (per-field state encoder + FiLM conditioning + spatial_inform)
# =============================================================================

class MultiFieldEncoder(nn.Module):
    """
    多场编码器 (同步了 physgto_v2.py 的 FiLM 条件化和 spatial_inform):
    - 每个物理场有独立的 state MLP
    - FiLM 风格调制：V = fv1(state) * γ + β，γ/β 由 time + condition + spatial 联合生成
    - 共享 edge 编码器
    """
    def __init__(self, space_size=3, n_fields=2, enc_dim=128,
                 enc_t_dim=11, cond_dim=32, spatial_dim=10):
        super().__init__()
        self.n_fields = n_fields

        # 每个场独立的 state encoder: input = 1 (单通道) + space_size
        self.fv_fields = nn.ModuleList([
            MLP(input_size=1 + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
            for _ in range(n_fields)
        ])

        # 共享的 time / condition / spatial encoder
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=cond_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_spatial = MLP(input_size=spatial_dim, output_size=enc_dim, act='SiLU', layer_norm=False)

        # FiLM 参数生成：[time, cond, spatial] -> (γ, β)
        self.fuse_para = MLP(input_size=enc_dim * 3, output_size=enc_dim * 2, act='SiLU', layer_norm=False)

        # 共享的 edge encoder
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges, spatial_inform):
        """
        Args:
            state_in:       [bs, N, n_fields]
            node_pos:       [bs, N, space_size]
            time_i:         [bs, enc_t_dim] — 时间编码（已 Fourier）
            conditions:     [bs, cond_dim]  — 原始条件值
            edges:          [bs, ne, 2]
            spatial_inform: [bs, spatial_dim] — 原始空间信息

        Returns:
            V_list: list of n_fields tensors, each [bs, N, enc_dim]
            E: [bs, ne, enc_dim]
        """
        time_enc    = self.fv_time(time_i)              # [bs, enc_dim]
        cond_enc    = self.fv_cond(conditions)           # [bs, enc_dim]
        spatial_enc = self.fv_spatial(spatial_inform)    # [bs, enc_dim]

        # FiLM 参数
        h = torch.cat([cond_enc, time_enc, spatial_enc], dim=-1)  # [bs, 3*enc_dim]
        para = self.fuse_para(h)
        gamma, beta = para.chunk(2, dim=-1)  # [bs, enc_dim] each

        V_list = []
        for i in range(self.n_fields):
            field_i = state_in[..., i:i+1]        # [bs, N, 1]
            inp = torch.cat([node_pos, field_i], dim=-1)
            V_i = self.fv_fields[i](inp) * gamma.unsqueeze(-2) + beta.unsqueeze(-2)
            V_list.append(V_i)

        E = self.fe(get_edge_info(edges, node_pos))
        return V_list, E


# =============================================================================
# AttnRes MixerBlock (per-field, with fine-grained AttnRes + Cross-Attention)
# =============================================================================

class AttnResMixerBlock(nn.Module):
    """
    融合 Block AttnRes + Multi-Field Cross-Attention 的 MixerBlock
    (使用 v2 增强版 GNN 和 Atten)
    """
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, n_fields=2,
                 cross_attn_heads=4, n_latent=4):
        super().__init__()
        self.n_fields = n_fields
        node_size = enc_dim + enc_s_dim

        # Per-field GNN (v2 注意力加权版)
        self.gnns = nn.ModuleList([
            GNN(node_size=node_size, edge_size=enc_dim, output_size=enc_dim, layer_norm=True)
            for _ in range(n_fields)
        ])

        # Cross-field attention (only when multiple fields exist)
        if n_fields > 1:
            self.cross_attns = nn.ModuleList([
                FieldCrossAttention(enc_dim, n_heads=cross_attn_heads, n_token=n_token)
                for _ in range(n_fields)
            ])
        else:
            self.cross_attns = None

        # Per-field Attention (v2 增强版：Xavier + q_offset + multi-latent)
        self.ln1s = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])
        self.mhas = nn.ModuleList([
            Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head, n_latent=n_latent)
            for _ in range(n_fields)
        ])

        # Per-field FFN
        self.ln2s = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in range(n_fields)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(enc_dim, 2 * enc_dim), nn.SiLU(), nn.Linear(2 * enc_dim, enc_dim))
            for _ in range(n_fields)
        ])

        # AttnRes: 每个场 × 3 个子层，初始化为零 → 等价于标准残差
        self.attn_res_w = nn.ParameterList([
            nn.Parameter(torch.zeros(enc_dim))
            for _ in range(n_fields * 3)
        ])
        self.attn_res_norm = RMSNorm()

    def _get_w(self, field_idx, sublayer_idx):
        return self.attn_res_w[field_idx * 3 + sublayer_idx]

    def forward(self, V_list, E_list, edges, s_enc, blocks_list):
        norm = self.attn_res_norm
        V_out = []
        E_out = []

        # ---- Step 1: Per-field GNN with AttnRes ----
        for i in range(self.n_fields):
            blocks_i = blocks_list[i]
            V_i = V_list[i]
            E_i = E_list[i]

            w_gnn = self._get_w(i, 0)
            h = block_attn_res(blocks_i, V_i, w_gnn, norm)

            V_in = torch.cat([h, s_enc], dim=-1)
            v, e = self.gnns[i](V_in, E_i, edges)
            E_i = E_i + e
            partial = v

            V_out.append(partial)
            E_out.append(E_i)

        # ---- Step 2: Cross-Field Attention ----
        if self.n_fields > 1:
            V_cross = []
            for i in range(self.n_fields):
                other_fields = [V_out[j] for j in range(self.n_fields) if j != i]
                if len(other_fields) == 1:
                    cross_info = self.cross_attns[i](V_out[i], other_fields[0])
                else:
                    other_cat = torch.cat(other_fields, dim=-2)
                    cross_info = self.cross_attns[i](V_out[i], other_cat)
                V_cross.append(V_out[i] + cross_info)
        else:
            V_cross = V_out

        # ---- Step 3: Per-field Attention with AttnRes ----
        V_attn = []
        for i in range(self.n_fields):
            blocks_i = blocks_list[i]
            partial = V_cross[i]

            w_attn = self._get_w(i, 1)
            h = block_attn_res(blocks_i, partial, w_attn, norm)

            attn_out = self.mhas[i](self.ln1s[i](h))
            partial = partial + attn_out
            V_attn.append(partial)

        # ---- Step 4: Per-field FFN with AttnRes ----
        V_final = []
        for i in range(self.n_fields):
            blocks_i = blocks_list[i]
            partial = V_attn[i]

            w_ffn = self._get_w(i, 2)
            h = block_attn_res(blocks_i, partial, w_ffn, norm)

            ffn_out = self.ffns[i](self.ln2s[i](h))
            partial = partial + ffn_out
            V_final.append(partial)

        # ---- End of block: append to blocks lists ----
        for i in range(self.n_fields):
            blocks_list[i] = blocks_list[i] + [V_final[i]]

        return V_final, E_out, blocks_list


# =============================================================================
# MultiFieldMixer
# =============================================================================

class MultiFieldMixer(nn.Module):
    def __init__(self, N_block, enc_dim, n_head, n_token, enc_s_dim, n_fields=2,
                 cross_attn_heads=4, n_latent=4):
        super().__init__()
        self.n_fields = n_fields
        self.blocks = nn.ModuleList([
            AttnResMixerBlock(
                enc_dim=enc_dim, n_head=n_head, n_token=n_token,
                enc_s_dim=enc_s_dim, n_fields=n_fields,
                cross_attn_heads=cross_attn_heads, n_latent=n_latent
            )
            for _ in range(N_block)
        ])

    def forward(self, V_list, E, edges, pos_enc):
        # 初始化 blocks_list: b_0 = encoder embedding for each field
        blocks_list = [[V_list[i]] for i in range(self.n_fields)]

        # 每个场复制一份 E 用于独立演化
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
# MultiFieldDecoder
# =============================================================================

class MultiFieldDecoder(nn.Module):
    """每个场独立的 Decoder (使用 v2 注意力加权版)"""
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
            delta_i = self.decoders[i](V_all_list[i], pos_enc)  # [bs, N, 1]
            deltas.append(delta_i)
        return torch.cat(deltas, dim=-1)  # [bs, N, n_fields]


# =============================================================================
# 完整模型
# =============================================================================

class Model(nn.Module):
    """
    PhysGTO-AttnRes-Multi-V3

    融合 Block Attention Residuals + Multi-Field Cross-Attention + V2 全部改进

    Args:
        space_size:       空间维度 (2 or 3)
        pos_enc_dim:      Fourier 编码频率数
        cond_dim:         原始条件维度（工艺参数）
        spatial_dim:      空间信息维度（坐标范围+网格数量等）
        N_block:          MixerBlock 数量
        in_dim:           输入通道数 (= n_fields)
        out_dim:          输出通道数 (= n_fields)
        enc_dim:          隐空间维度
        n_head:           Attention 头数
        n_token:          投影注意力查询数
        n_latent:         Atten 内部 latent self-attention 层数
        dt:               默认时间步
        stepper_scheme:   "euler" 或 "delta"
        pos_x_boost:      x 轴位置编码频率 boost
        n_fields:         物理场数量
        cross_attn_heads: Cross-Attention 头数
    """
    def __init__(self,
                 space_size=3,
                 pos_enc_dim=5,
                 cond_dim=32,
                 spatial_dim=10,
                 N_block=4,
                 in_dim=2,
                 out_dim=2,
                 enc_dim=128,
                 n_head=4,
                 n_token=128,
                 n_latent=4,
                 dt: float = 0.05,
                 stepper_scheme="euler",
                 pos_x_boost=2,
                 n_fields=None,
                 cross_attn_heads=4,
                 ):
        super().__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.pos_enc_dim = pos_enc_dim
        self.pos_x_boost = pos_x_boost
        self.n_fields = n_fields if n_fields is not None else in_dim

        # enc_s_dim: 各向异性 Fourier，输出维度
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size

        # enc_t_dim: [t, dt] Fourier(dim=pos_enc_dim) + 单独 t 低频 Fourier(dim=2)
        # FourierEmbedding([t, dt]) -> 2 + 4*pos_enc_dim
        # FourierEmbedding_lowfreq(t)  -> 1 + 4 = 5
        enc_t_dim = (1 + 2 * pos_enc_dim) * 2 + (1 + 2 * 2)

        self.encoder = MultiFieldEncoder(
            space_size=space_size,
            n_fields=self.n_fields,
            enc_dim=enc_dim,
            enc_t_dim=enc_t_dim,
            cond_dim=cond_dim,
            spatial_dim=spatial_dim,
        )

        self.mixer = MultiFieldMixer(
            N_block=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
            cross_attn_heads=cross_attn_heads,
            n_latent=n_latent,
        )

        self.decoder = MultiFieldDecoder(
            N_block=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
        )

    def _encode_time(self, time_i, dt_tensor):
        """时间编码：[t, dt] Fourier + t 低频 Fourier(dim=2)"""
        time_info = torch.cat([time_i, dt_tensor], dim=-1)            # (bs, 2)
        t_fourier = FourierEmbedding(time_info, 0, self.pos_enc_dim)  # (bs, 2+4*pos_enc_dim)
        t_low = FourierEmbedding(time_i, 0, 2)                        # (bs, 1+4) = (bs, 5)
        return torch.cat([t_fourier, t_low], dim=-1)                   # (bs, enc_t_dim)

    def forward(self, state_in, node_pos, edges, time_i, conditions, spatial_inform,
                pos_enc=None, dt=None):
        """
        单步预测

        Args:
            state_in:       [bs, N, in_dim]
            node_pos:       [bs, N, space_size]
            edges:          [bs, ne, 2]
            time_i:         [bs,] or [bs, 1]
            conditions:     [bs, cond_dim]      — 原始条件值
            spatial_inform: [bs, spatial_dim]   — 原始空间信息
            pos_enc:        [bs, N, enc_s_dim]  — 可选，外部预计算
            dt:             float/tensor/None

        Returns:
            state_pred: [bs, N, out_dim]
        """
        if pos_enc is None:
            pos_enc = FourierEmbedding_pos(node_pos, self.pos_enc_dim, self.pos_x_boost)

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

        t_enc = self._encode_time(time_i, dt_tensor)

        edges_long = edges.long() if edges.dtype != torch.long else edges

        # Encoder: 多场 FiLM 编码
        V_list, E = self.encoder(node_pos, state_in, t_enc, conditions, edges_long, spatial_inform)

        # Mixer: Block AttnRes + Cross-Attention
        V_all_list = self.mixer(V_list, E, edges_long, pos_enc)

        # Decoder: 注意力加权 block 聚合
        v_pred = self.decoder(V_all_list, pos_enc)

        if self.stepper_scheme == "euler":
            with autocast(device_type="cuda", enabled=False):
                state_pred = state_in.float() + v_pred.float() * dt_tensor.unsqueeze(-1).float()
        else:
            state_pred = state_in + v_pred

        return state_pred

    def autoregressive(self,
                       state_in,
                       node_pos,
                       edges,
                       time_seq,
                       spatial_inform,
                       conditions,
                       dt=None,
                       check_point=False,
                       teacher_forcing=False,
                       gt_states=None):
        """
        自回归多步预测

        Args:
            state_in:       [bs, N, in_dim]
            node_pos:       [bs, N, space_size]
            edges:          [bs, ne, 2]
            time_seq:       [bs, T] or [bs, T, 1]
            spatial_inform: [bs, spatial_dim]
            conditions:     [bs, cond_dim]
            dt:             float/tensor/None
            check_point:    bool
            teacher_forcing: bool
            gt_states:      [bs, T, N, in_dim] or None

        Returns:
            outputs: [bs, T, N, out_dim]
        """
        state_t = state_in
        outputs = [state_in]
        T = time_seq.shape[1]

        pos_enc = FourierEmbedding_pos(node_pos, self.pos_enc_dim, self.pos_x_boost)

        for t in range(T):
            time_i = time_seq[:, t]

            def custom_forward(s_t, t_i):
                return self.forward(s_t, node_pos, edges, t_i, conditions, spatial_inform, pos_enc, dt)

            if check_point:
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t = state_t.detach().requires_grad_(True)
                state_pred = checkpoint(custom_forward, state_t, time_i, use_reentrant=False)
            else:
                state_pred = self.forward(state_t, node_pos, edges, time_i, conditions, spatial_inform, pos_enc, dt)

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
    print("PhysGTO-AttnRes-Multi-V3 快速验证")
    print("=" * 60)

    bs, N, ne = 2, 64, 128
    T = 4
    space_dim = 3
    in_dim = out_dim = 2   # T + alpha.air
    cond_dim = 8
    spatial_dim = 10

    model = Model(
        space_size=space_dim,
        pos_enc_dim=3,
        cond_dim=cond_dim,
        spatial_dim=spatial_dim,
        N_block=2,
        in_dim=in_dim,
        out_dim=out_dim,
        enc_dim=64,
        n_head=4,
        n_token=32,
        n_latent=2,
        dt=2e-5,
        n_fields=2,
        cross_attn_heads=4,
        pos_x_boost=2,
    )

    state_in = torch.randn(bs, N, in_dim)
    node_pos = torch.rand(bs, N, space_dim)
    edges = torch.randint(0, N, (bs, ne, 2))
    time_seq = torch.linspace(0, 1e-4, T).unsqueeze(0).expand(bs, -1)
    conditions = torch.randn(bs, cond_dim)
    spatial_inform = torch.randn(bs, spatial_dim)

    # 单步
    pred = model(state_in, node_pos, edges, time_seq[:, 0], conditions, spatial_inform)
    print(f"[单步]  pred: {pred.shape}")
    assert pred.shape == (bs, N, out_dim)

    # 自回归
    out = model.autoregressive(state_in, node_pos, edges, time_seq, spatial_inform, conditions)
    print(f"[自回归] out: {out.shape}")
    assert out.shape == (bs, T, N, out_dim)

    # 参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {params/1e6:.3f}M")

    # 检查 AttnRes pseudo-query 初始化
    for name, p in model.named_parameters():
        if 'attn_res_w' in name:
            assert torch.all(p == 0), f"{name} should be initialized to zero!"
    print("AttnRes pseudo-query 全部初始化为零 ✓")

    print("\n✅  PhysGTO-AttnRes-Multi-V3 双场验证通过！")

    # ---- 单场测试 ----
    print(f"\n{'─'*40}")
    print("  Testing n_fields=1 (single field)")
    print(f"{'─'*40}")

    model1 = Model(
        space_size=space_dim,
        pos_enc_dim=3,
        cond_dim=cond_dim,
        spatial_dim=spatial_dim,
        N_block=2,
        in_dim=1,
        out_dim=1,
        enc_dim=64,
        n_head=4,
        n_token=32,
        n_latent=2,
        dt=2e-5,
        n_fields=1,
        cross_attn_heads=4,
    )

    state_in_1 = torch.randn(bs, N, 1)
    pred1 = model1(state_in_1, node_pos, edges, time_seq[:, 0], conditions, spatial_inform)
    print(f"[单步]  pred: {pred1.shape}")
    assert pred1.shape == (bs, N, 1)

    out1 = model1.autoregressive(state_in_1, node_pos, edges, time_seq, spatial_inform, conditions)
    print(f"[自回归] out: {out1.shape}")
    assert out1.shape == (bs, T, N, 1)

    model1.train()
    out1_ck = model1.autoregressive(state_in_1, node_pos, edges, time_seq, spatial_inform, conditions, check_point=True)
    loss1 = out1_ck.sum()
    loss1.backward()
    print(f"[checkpoint] backward pass ✓")

    params1 = sum(p.numel() for p in model1.parameters())
    print(f"参数量: {params1/1e6:.3f}M (no cross-attention modules)")
    assert not hasattr(model1.mixer.blocks[0], 'cross_attns') or model1.mixer.blocks[0].cross_attns is None
    print("Cross-attention correctly skipped for n_fields=1 ✓")

    print("\n✅  PhysGTO-AttnRes-Multi-V3 单场 + 双场全部验证通过！")
