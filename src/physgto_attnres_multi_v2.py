"""
PhysGTO-AttnRes-Multi v2: Block Attention Residuals + Multi-Field Cross-Attention

v2 改进要点（对比 v1）：
================================================================================
1. AttnRes 模式可选 (attn_res_mode 参数)
   - "block_inter" (默认, 论文推荐):
     Block 间采用 AttnRes（每个 block 开头 1 次），Block 内采用标准残差
     使用 GatedCrossAttention (单步 + 门控)
   - "full" (v1 风格):
     每个子层前都做 AttnRes (3次/block/field)
     使用 FieldCrossAttention (3步 Projection-Inspired, 线性复杂度)

2. Cross-Attention 两种方案
   - block_inter 模式: GatedCrossAttention (单步 + 门控, 参数少)
   - full 模式: FieldCrossAttention (3步 Projection-Inspired)

3. 残差流完整性
   - block_inter: block_attn_res → h = h + gnn → h = h + cross → h = h + attn → h = h + ffn
   - full: 每子层前 block_attn_res, partial 只累加子层输出

4. Encoder 增强：场间交互 embedding (门控加法融合)

5. 配置方式
   - Model(..., attn_res_mode="block_inter")  # 论文推荐
   - Model(..., attn_res_mode="full")          # v1 风格
   - JSON config: "attn_res_mode": "block_inter" 或 "full"
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np

from torch.amp import GradScaler, autocast
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


# =============================================================================
# AttnRes 核心组件
# =============================================================================

class RMSNorm(nn.Module):
    """Parameter-free RMSNorm (论文 Eq.2 中 φ(q,k) = exp(q^T RMSNorm(k)))"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


def block_attn_res(blocks, partial_block, w, rms_norm):
    """
    Block Attention Residuals (论文 Section 3.2, Figure 2 的 block_attn_res 函数)

    严格还原论文公式:
        V = [b_0, b_1, ..., b_{n-1}, partial_block]  (所有 source)
        K = RMSNorm(V)                                 (key 归一化)
        α_{i→l} = softmax(w_l^T · K_i)               (深度注意力权重)
        h_l = Σ α_{i→l} · V_i                        (加权聚合)

    Args:
        blocks: list of [bs, N, D] — 已完成的 block 表示 [b_0, ..., b_{n-1}]
        partial_block: [bs, N, D] — 当前 block 内的部分累加
        w: nn.Parameter [D] — 伪查询向量 (零初始化 → 初始均匀注意力)
        rms_norm: RMSNorm — 对 key 归一化防止大幅值层主导

    Returns:
        h: [bs, N, D] — 深度加权聚合结果
    """
    sources = blocks + [partial_block]
    V = torch.stack(sources, dim=0)                    # [S, bs, N, D]
    K = rms_norm(V)                                     # RMSNorm on keys
    logits = torch.einsum('d, s b n d -> s b n', w, K)  # [S, bs, N]
    alpha = logits.softmax(dim=0)                       # softmax over sources
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
    """Projection-Inspired Attention: 三步 Q→W0, W→W, W0→W"""
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

    def forward(self, V, E, edges):
        bs, N, _ = V.shape
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
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


class GatedCrossAttention(nn.Module):
    """
    门控 Projection-Inspired Cross-Field Attention (线性复杂度)

    结合 v1 的 Projection-Inspired 方案 + 门控渐进耦合：
    - 使用学习的 query token 作为中介，复杂度 O(N × n_token) 而非 O(N²)
    - 添加可学习标量门控 gate（初始化为 0）
    - 训练初期 gate≈0 → cross-attention 不干扰主分支
    - 训练后期 gate 逐渐增大 → 逐步引入跨场耦合
    """
    def __init__(self, enc_dim, n_heads=4, n_token=64):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(n_token, enc_dim))
        self.ln_other = nn.LayerNorm(enc_dim)
        self.ln_self = nn.LayerNorm(enc_dim)
        self.attn1 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        self.attn3 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        # 门控初始化为 0 → 训练开始时不影响主分支
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, V_self, V_other):
        """
        V_self:  [bs, N, D] — 当前场
        V_other: [bs, N, D] — 另一个场
        Returns: [bs, N, D] — 门控后的跨场信息 (线性复杂度)
        """
        bs = V_self.shape[0]
        Q = self.Q.unsqueeze(0).expand(bs, -1, -1)
        other = self.ln_other(V_other)
        self_normed = self.ln_self(V_self)
        W, _ = self.attn1(Q, other, other)       # [bs, n_token, D]
        W, _ = self.attn2(W, W, W)               # [bs, n_token, D]
        out, _ = self.attn3(self_normed, W, W)    # [bs, N, D]
        return torch.tanh(self.gate) * out


class FieldCrossAttention(nn.Module):
    """
    Projection-Inspired Cross-Field Attention (线性复杂度, 来自 v1)

    用于 attn_res_mode="full" 时的场间耦合：
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
        bs = V_self.shape[0]
        Q = self.Q.unsqueeze(0).expand(bs, -1, -1)
        other = self.ln_other(V_other)
        self_normed = self.ln_self(V_self)
        W, _ = self.attn1(Q, other, other)
        W, _ = self.attn2(W, W, W)
        out, _ = self.attn3(self_normed, W, W)
        return out


# =============================================================================
# Decoder (per-field)
# =============================================================================

class Decoder(nn.Module):
    def __init__(self, N=4, enc_dim=128, enc_s_dim=10, state_size=1):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(N * enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc):
        b, n_block, N, enc_dim = V_all.shape
        V_all = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        V = self.delta_net(torch.cat([V_all, pos_enc], dim=-1))
        return V


# =============================================================================
# Encoder (per-field + 场间信息交换)
# =============================================================================

class MultiFieldEncoder(nn.Module):
    """
    多场编码器 v2：
    - 每个物理场有独立的 state MLP
    - 共享 time / condition / edge 编码器
    - 新增：轻量级场间信息交换（门控加法融合）
    """
    def __init__(self, space_size=3, n_fields=2, enc_dim=128,
                 enc_t_dim=11, enc_c_dim=12):
        super().__init__()
        self.n_fields = n_fields

        # 每个场独立的 state encoder: input = 1 (单通道) + space_size
        self.fv_fields = nn.ModuleList([
            MLP(input_size=1 + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
            for _ in range(n_fields)
        ])

        # 共享的 time / condition encoder
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim, act='SiLU', layer_norm=False)

        # 共享的 edge encoder
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

        # 场间信息交换：将其他场的信息通过门控加入
        # 每个场有一个投影层 + 门控
        self.field_exchange = nn.ModuleList([
            nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.SiLU())
            for _ in range(n_fields)
        ])
        self.field_exchange_gate = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_fields)
        ])

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        time_enc = self.fv_time(time_i)
        cond_enc = self.fv_cond(conditions)

        V_list = []
        for i in range(self.n_fields):
            field_i = state_in[..., i:i+1]
            inp = torch.cat([field_i, node_pos], dim=-1)
            V_i = self.fv_fields[i](inp) + time_enc.unsqueeze(-2) + cond_enc.unsqueeze(-2)
            V_list.append(V_i)

        # 场间信息交换：将其他场的 embedding 通过门控加入
        V_exchanged = []
        for i in range(self.n_fields):
            other_sum = sum(V_list[j] for j in range(self.n_fields) if j != i)
            exchange_info = self.field_exchange[i](other_sum)
            gate = torch.tanh(self.field_exchange_gate[i])
            V_exchanged.append(V_list[i] + gate * exchange_info)

        E = self.fe(get_edge_info(edges, node_pos))
        return V_exchanged, E


# =============================================================================
# AttnRes MixerBlock v2 (严格还原论文 Figure 2)
# =============================================================================

class AttnResMixerBlock(nn.Module):
    """
    统一 MixerBlock：支持两种 AttnRes 模式

    attn_res_mode:
    ─────────────────────────────────────────
    "block_inter" (默认, 论文推荐):
      - Block 开头 1 次 AttnRes → 聚合所有历史 block
      - Block 内部全部标准残差 (h = h + sublayer_out)
      - 使用 GatedCrossAttention (单步 + 门控)
      - 计算高效，忠于论文 Figure 2

    "full" (v1 风格):
      - 每个子层前都做 AttnRes (3次/block/field)
      - 使用 FieldCrossAttention (3步 Projection-Inspired)
      - partial_block 只累加子层输出，AttnRes 分别在 GNN/Attn/FFN 前应用
    ─────────────────────────────────────────
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

        # Per-field Cross-Field Attention
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
            # 每个场 × 3 个子层 = 3 * n_fields 个 pseudo-query
            self.attn_res_w = nn.ParameterList([
                nn.Parameter(torch.zeros(enc_dim))
                for _ in range(n_fields * 3)
            ])
        else:
            # 每个场 1 个 pseudo-query (仅 block 间)
            self.attn_res_w = nn.ParameterList([
                nn.Parameter(torch.zeros(enc_dim))
                for _ in range(n_fields)
            ])
        self.attn_res_norm = RMSNorm()

    def _get_w(self, field_idx, sublayer_idx):
        """获取第 field_idx 场、第 sublayer_idx 子层的 pseudo-query (full mode)"""
        return self.attn_res_w[field_idx * 3 + sublayer_idx]

    def _forward_block_inter(self, V_list, E_list, edges, s_enc, blocks_list):
        """block_inter 模式: AttnRes 仅在 block 开头, block 内标准残差"""
        norm = self.attn_res_norm
        V_out = []
        E_out = []

        # ---- Step 1: Per-field Block间 AttnRes + GNN (标准残差) ----
        for i in range(self.n_fields):
            blocks_i = blocks_list[i]
            V_i = V_list[i]
            E_i = E_list[i]

            w = self.attn_res_w[i]
            h = block_attn_res(blocks_i, V_i, w, norm)

            V_in = torch.cat([h, s_enc], dim=-1)
            v, e = self.gnns[i](V_in, E_i, edges)
            E_i = E_i + e
            h = h + v

            V_out.append(h)
            E_out.append(E_i)

        # ---- Step 2: Cross-Field Attention（门控残差）----
        V_cross = []
        for i in range(self.n_fields):
            other_fields = [V_out[j] for j in range(self.n_fields) if j != i]
            if len(other_fields) == 1:
                cross_info = self.cross_attns[i](V_out[i], other_fields[0])
            else:
                other_cat = torch.cat(other_fields, dim=-2)
                cross_info = self.cross_attns[i](V_out[i], other_cat)
            V_cross.append(V_out[i] + cross_info)

        # ---- Step 3: Per-field Attention（标准残差）----
        V_attn = []
        for i in range(self.n_fields):
            h = V_cross[i]
            h = h + self.mhas[i](self.ln1s[i](h))
            V_attn.append(h)

        # ---- Step 4: Per-field FFN（标准残差）----
        V_final = []
        for i in range(self.n_fields):
            h = V_attn[i]
            h = h + self.ffns[i](self.ln2s[i](h))
            V_final.append(h)

        # ---- Block 结束：append 到 blocks_list ----
        for i in range(self.n_fields):
            blocks_list[i] = blocks_list[i] + [V_final[i]]

        return V_final, E_out, blocks_list

    def _forward_full(self, V_list, E_list, edges, s_enc, blocks_list):
        """full 模式: 每个子层前都做 AttnRes (v1 风格)"""
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
            partial = h + v  # 保留完整残差（h 是 AttnRes 聚合结果）

            V_out.append(partial)
            E_out.append(E_i)

        # ---- Step 2: Cross-Field Attention ----
        V_cross = []
        for i in range(self.n_fields):
            other_fields = [V_out[j] for j in range(self.n_fields) if j != i]
            if len(other_fields) == 1:
                cross_info = self.cross_attns[i](V_out[i], other_fields[0])
            else:
                other_cat = torch.cat(other_fields, dim=-2)
                cross_info = self.cross_attns[i](V_out[i], other_cat)
            V_cross.append(V_out[i] + cross_info)

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

    def forward(self, V_list, E_list, edges, s_enc, blocks_list):
        if self.attn_res_mode == "full":
            return self._forward_full(V_list, E_list, edges, s_enc, blocks_list)
        else:
            return self._forward_block_inter(V_list, E_list, edges, s_enc, blocks_list)


# =============================================================================
# MultiFieldMixer
# =============================================================================

class MultiFieldMixer(nn.Module):
    def __init__(self, N_block, enc_dim, n_head, n_token, enc_s_dim, n_fields=2,
                 cross_attn_heads=4, attn_res_mode="block_inter"):
        super().__init__()
        self.n_fields = n_fields
        self.blocks = nn.ModuleList([
            AttnResMixerBlock(
                enc_dim=enc_dim, n_head=n_head, n_token=n_token,
                enc_s_dim=enc_s_dim, n_fields=n_fields, cross_attn_heads=cross_attn_heads,
                attn_res_mode=attn_res_mode
            )
            for _ in range(N_block)
        ])

    def forward(self, V_list, E, edges, pos_enc):
        # b_0 = encoder embedding
        blocks_list = [[V_list[i]] for i in range(self.n_fields)]

        # 各场独立演化边特征
        E_list = [E.clone() for _ in range(self.n_fields)]

        V_all = [[] for _ in range(self.n_fields)]

        for block in self.blocks:
            V_list, E_list, blocks_list = block(V_list, E_list, edges, pos_enc, blocks_list)
            for i in range(self.n_fields):
                V_all[i].append(V_list[i])

        V_all_stacked = [torch.stack(V_all[i], dim=1) for i in range(self.n_fields)]
        return V_all_stacked


# =============================================================================
# MultiFieldDecoder
# =============================================================================

class MultiFieldDecoder(nn.Module):
    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10, n_fields=2):
        super().__init__()
        self.n_fields = n_fields
        self.decoders = nn.ModuleList([
            Decoder(N=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, state_size=1)
            for _ in range(n_fields)
        ])

    def forward(self, V_all_list, pos_enc):
        deltas = []
        for i in range(self.n_fields):
            delta_i = self.decoders[i](V_all_list[i], pos_enc)
            deltas.append(delta_i)
        return torch.cat(deltas, dim=-1)


# =============================================================================
# 完整模型
# =============================================================================

class Model(nn.Module):
    """
    PhysGTO-AttnRes-Multi v2

    改进要点：
    1. Block间 AttnRes（1次/block） + Block内标准残差 → 计算高效且忠于论文
    2. 门控 Cross-Attention → 训练初期不干扰，后期渐进耦合
    3. Encoder 场间信息交换 → 初始 embedding 包含多场信息
    4. 接口完全兼容 physgto_res.py
    5. attn_res_mode 开关: "block_inter"(默认) / "full"(每子层AttnRes)
    """
    def __init__(self,
                 space_size=3,
                 pos_enc_dim=5,
                 cond_dim=32,
                 N_block=4,
                 in_dim=2,
                 out_dim=2,
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
        enc_t_dim = 2 * (1 + 2 * pos_enc_dim)
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.encoder = MultiFieldEncoder(
            space_size=space_size,
            n_fields=self.n_fields,
            enc_dim=enc_dim,
            enc_t_dim=enc_t_dim,
            enc_c_dim=enc_c_dim,
        )

        self.mixer = MultiFieldMixer(
            N_block=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
            cross_attn_heads=cross_attn_heads,
            attn_res_mode=attn_res_mode,
        )

        self.decoder = MultiFieldDecoder(
            N_block=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
        )

    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
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

        V_list, E = self.encoder(node_pos, state_in, t_enc, c_enc, edges_long)
        V_all_list = self.mixer(V_list, E, edges_long, pos_enc)
        v_pred = self.decoder(V_all_list, pos_enc)

        # with autocast(device_type="cuda", enabled=False):
        #     state_pred = state_in.float() + v_pred.float() * dt_tensor.float()
            
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
    print("PhysGTO-AttnRes-Multi v2 快速验证 (双模式)")
    print("=" * 60)

    bs, N, ne = 2, 64, 128
    T = 4
    space_dim = 3
    in_dim = out_dim = 2
    cond_dim = 8

    for mode in ["block_inter", "full"]:
        print(f"\n{'─'*40}")
        print(f"  Testing mode: {mode}")
        print(f"{'─'*40}")

        model = Model(
            space_size=space_dim,
            pos_enc_dim=3,
            cond_dim=cond_dim,
            N_block=4,
            in_dim=in_dim,
            out_dim=out_dim,
            enc_dim=64,
            n_head=4,
            n_token=32,
            dt=2e-5,
            n_fields=2,
            cross_attn_heads=4,
            attn_res_mode=mode,
        )

        state_in = torch.randn(bs, N, in_dim)
        node_pos = torch.rand(bs, N, space_dim)
        edges = torch.randint(0, N, (bs, ne, 2))
        time_seq = torch.linspace(0, 1e-4, T).unsqueeze(0).expand(bs, -1)
        conditions = torch.randn(bs, cond_dim)

        # 单步
        pred = model(state_in, node_pos, edges, time_seq[:, 0], conditions)
        print(f"[单步]  pred: {pred.shape}")
        assert pred.shape == (bs, N, out_dim)

        # 自回归
        out = model.autoregressive(state_in, node_pos, edges, time_seq, conditions)
        print(f"[自回归] out: {out.shape}")
        assert out.shape == (bs, T, N, out_dim)

        # checkpoint backward
        model.train()
        out_ck = model.autoregressive(state_in, node_pos, edges, time_seq, conditions, check_point=True)
        loss = out_ck.sum()
        loss.backward()
        print(f"[checkpoint] backward pass ✓")

        # 参数量
        total = sum(p.numel() for p in model.parameters())
        attnres_params = sum(p.numel() for n, p in model.named_parameters() if 'attn_res' in n)
        gate_params = sum(p.numel() for n, p in model.named_parameters() if 'gate' in n)
        print(f"参数量: {total/1e6:.3f}M (AttnRes: {attnres_params}, Gates: {gate_params})")

        # 零初始化
        for name, p in model.named_parameters():
            if 'attn_res_w' in name:
                assert torch.all(p == 0), f"{name} not zero!"
        print("AttnRes pseudo-query 零初始化 ✓")

        if mode == "block_inter":
            for name, p in model.named_parameters():
                if 'gate' in name:
                    assert torch.all(p == 0), f"{name} not zero!"
            print("Gate 零初始化 ✓")

        print(f"✅  mode='{mode}' 验证通过！")

    print(f"\n{'='*60}")
    print("✅  PhysGTO-AttnRes-Multi v2 双模式全部验证通过！")
    print(f"{'='*60}")
