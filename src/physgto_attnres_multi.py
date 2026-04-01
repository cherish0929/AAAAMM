"""
PhysGTO-AttnRes-Multi: Block Attention Residuals + Multi-Field Cross-Attention

基于 PhysGTO-Res (physgto_res.py) 的两项核心改进：

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


# =============================================================================
# Node-wise Latent Memory Block (GRU-style gated recurrent)
# =============================================================================

class LatentMemoryBlock(nn.Module):
    """
    Node-wise GRU-style 门控 Memory 更新模块

    memory 维度: [B, N, mem_dim]
    在 Encoder 输出后、Mixer 前 对每个节点独立做门控更新:
        z = σ(W_z [h_enc; mem])     — update gate
        r = σ(W_r [h_enc; mem])     — reset gate
        m̃ = tanh(W_m [h_enc; r ⊙ mem])  — candidate
        mem_new = (1 - z) ⊙ mem + z ⊙ m̃
    """
    def __init__(self, enc_dim, mem_dim=None):
        super().__init__()
        mem_dim = mem_dim or enc_dim
        self.mem_dim = mem_dim
        self.enc_dim = enc_dim

        cat_dim = enc_dim + mem_dim
        self.W_z = nn.Linear(cat_dim, mem_dim)
        self.W_r = nn.Linear(cat_dim, mem_dim)
        self.W_m = nn.Linear(enc_dim + mem_dim, mem_dim)
        # 投影: 当 mem_dim != enc_dim 时，将 memory 映射回 enc_dim
        self.proj = nn.Linear(mem_dim, enc_dim) if mem_dim != enc_dim else nn.Identity()

    def forward(self, h_enc, memory):
        """
        Args:
            h_enc:  [B, N, enc_dim] — Encoder 输出
            memory: [B, N, mem_dim] — 上一步 memory (若 None 则初始化为零)
        Returns:
            h_fused: [B, N, enc_dim] — 融合 memory 后的 Encoder 表示 (送入 Mixer)
            mem_new: [B, N, mem_dim] — 更新后的 memory
        """
        if memory is None:
            memory = torch.zeros(
                h_enc.shape[0], h_enc.shape[1], self.mem_dim,
                device=h_enc.device, dtype=h_enc.dtype
            )

        cat = torch.cat([h_enc, memory], dim=-1)  # [B, N, enc_dim + mem_dim]
        z = torch.sigmoid(self.W_z(cat))
        r = torch.sigmoid(self.W_r(cat))

        cat_r = torch.cat([h_enc, r * memory], dim=-1)
        m_cand = torch.tanh(self.W_m(cat_r))

        mem_new = (1.0 - z) * memory + z * m_cand

        # 融合: 将 memory 信息注入 Encoder 表示
        h_fused = h_enc + self.proj(mem_new)
        return h_fused, mem_new

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
        # Step 1: learned tokens attend to other field
        self.attn1 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        # Step 2: tokens self-refine
        self.attn2 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)
        # Step 3: self field attend to refined tokens
        self.attn3 = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True)

    def forward(self, V_self, V_other):
        """
        V_self:  [bs, N, enc_dim] — 当前场节点特征
        V_other: [bs, N, enc_dim] — 另一个场节点特征
        Returns: [bs, N, enc_dim] — 从其他场获取的跨场信息
        """
        bs = V_self.shape[0]
        Q = self.Q.unsqueeze(0).expand(bs, -1, -1)         # [bs, n_token, D]
        other = self.ln_other(V_other)                       # [bs, N, D]
        self_normed = self.ln_self(V_self)                   # [bs, N, D]

        W, _ = self.attn1(Q, other, other)                   # [bs, n_token, D]
        W, _ = self.attn2(W, W, W)                           # [bs, n_token, D]
        out, _ = self.attn3(self_normed, W, W)               # [bs, N, D]
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
        # V_all: [bs, n_block, N, enc_dim]
        b, n_block, N, enc_dim = V_all.shape
        V_all = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        V = self.delta_net(torch.cat([V_all, pos_enc], dim=-1))
        return V


# =============================================================================
# Encoder (per-field state encoder + shared edge/time/cond encoders)
# =============================================================================

class MultiFieldEncoder(nn.Module):
    """
    多场编码器：
    - 每个物理场有独立的 state MLP (fv_field_i)
    - 共享 time / condition / edge 编码器
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

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        """
        Args:
            state_in: [bs, N, n_fields] — 所有场拼接
            node_pos: [bs, N, space_size]
            time_i: [bs, enc_t_dim] — 时间编码
            conditions: [bs, enc_c_dim] — 条件编码
            edges: [bs, ne, 2]

        Returns:
            V_list: list of n_fields tensors, each [bs, N, enc_dim]
            E: [bs, ne, enc_dim] — 共享初始边特征
        """
        time_enc = self.fv_time(time_i)           # [bs, enc_dim]
        cond_enc = self.fv_cond(conditions)       # [bs, enc_dim]

        V_list = []
        for i in range(self.n_fields):
            field_i = state_in[..., i:i+1]        # [bs, N, 1]
            inp = torch.cat([field_i, node_pos], dim=-1)  # [bs, N, 1+space]
            V_i = self.fv_fields[i](inp) + time_enc.unsqueeze(-2) + cond_enc.unsqueeze(-2)
            V_list.append(V_i)

        E = self.fe(get_edge_info(edges, node_pos))
        return V_list, E


# =============================================================================
# AttnRes MixerBlock (per-field, with fine-grained AttnRes + Cross-Attention)
# =============================================================================

class AttnResMixerBlock(nn.Module):
    """
    融合 Block AttnRes + Multi-Field Cross-Attention 的 MixerBlock

    每个物理场有独立的 GNN / Attention / FFN；
    场间通过 FieldCrossAttention 耦合；
    每个子层前插入 block_attn_res。

    AttnRes 在每个子层前应用（细粒度，3次/block）：
    - Before GNN: h = block_attn_res(blocks, partial, w_gnn, norm)
    - Before Attention: h = block_attn_res(blocks, partial, w_attn, norm)
    - Before FFN: h = block_attn_res(blocks, partial, w_ffn, norm)
    """
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim, n_fields=2, cross_attn_heads=4):
        super().__init__()
        self.n_fields = n_fields
        node_size = enc_dim + enc_s_dim

        # Per-field GNN
        self.gnns = nn.ModuleList([
            GNN(node_size=node_size, edge_size=enc_dim, output_size=enc_dim, layer_norm=True)
            for _ in range(n_fields)
        ])

        # Cross-field attention (双向: field_i → field_j 和 field_j → field_i)
        # 使用 Projection-Inspired 方式，避免 O(N²) 复杂度
        self.cross_attns = nn.ModuleList([
            FieldCrossAttention(enc_dim, n_heads=cross_attn_heads, n_token=n_token)
            for _ in range(n_fields)  # cross_attns[i] 让 field_i attend to 其他场
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

        # AttnRes: 每个场 × 3 个子层 = 3 * n_fields 个 pseudo-query
        # 初始化为零 → 初始时等价于标准残差
        self.attn_res_w = nn.ParameterList([
            nn.Parameter(torch.zeros(enc_dim))     # w_{field_i, sublayer_j}
            for _ in range(n_fields * 3)
        ])
        self.attn_res_norm = RMSNorm()

    def _get_w(self, field_idx, sublayer_idx):
        """获取第 field_idx 场、第 sublayer_idx 子层的 pseudo-query"""
        return self.attn_res_w[field_idx * 3 + sublayer_idx]

    def forward(self, V_list, E_list, edges, s_enc, blocks_list):
        """
        Args:
            V_list: list of [bs, N, enc_dim] per field
            E_list: list of [bs, ne, enc_dim] per field (各场独立演化)
            edges: [bs, ne, 2]
            s_enc: [bs, N, enc_s_dim] — 空间位置编码
            blocks_list: list of (list of [bs, N, enc_dim]) per field — 每场的历史 block 列表

        Returns:
            V_list_out, E_list_out, blocks_list (updated)
        """
        norm = self.attn_res_norm
        V_out = []
        E_out = []

        # ---- Step 1: Per-field GNN with AttnRes ----
        for i in range(self.n_fields):
            blocks_i = blocks_list[i]
            V_i = V_list[i]
            E_i = E_list[i]

            # AttnRes before GNN
            w_gnn = self._get_w(i, 0)
            h = block_attn_res(blocks_i, V_i, w_gnn, norm)

            # GNN
            V_in = torch.cat([h, s_enc], dim=-1)
            v, e = self.gnns[i](V_in, E_i, edges)
            E_i = E_i + e
            # partial_block starts as GNN output (first sublayer in block)
            partial = v

            V_out.append(partial)
            E_out.append(E_i)

        # ---- Step 2: Cross-Field Attention ----
        # 对 n_fields=2: V_0 attend to V_1, V_1 attend to V_0
        V_cross = []
        for i in range(self.n_fields):
            # 聚合来自其他所有场的信息
            other_fields = [V_out[j] for j in range(self.n_fields) if j != i]
            # 对于 2 场，only 1 other; 对于更多场可以 concatenate 或逐个做
            if len(other_fields) == 1:
                cross_info = self.cross_attns[i](V_out[i], other_fields[0])
            else:
                # 多场：concatenate 其他场的 token 序列
                other_cat = torch.cat(other_fields, dim=-2)  # [bs, N*(n_fields-1), enc_dim]
                cross_info = self.cross_attns[i](V_out[i], other_cat)
            V_cross.append(V_out[i] + cross_info)

        # ---- Step 3: Per-field Attention with AttnRes ----
        V_attn = []
        for i in range(self.n_fields):
            blocks_i = blocks_list[i]
            partial = V_cross[i]

            # AttnRes before Attention
            w_attn = self._get_w(i, 1)
            h = block_attn_res(blocks_i, partial, w_attn, norm)

            # Attention (PreNorm + residual to partial)
            attn_out = self.mhas[i](self.ln1s[i](h))
            partial = partial + attn_out
            V_attn.append(partial)

        # ---- Step 4: Per-field FFN with AttnRes ----
        V_final = []
        for i in range(self.n_fields):
            blocks_i = blocks_list[i]
            partial = V_attn[i]

            # AttnRes before FFN
            w_ffn = self._get_w(i, 2)
            h = block_attn_res(blocks_i, partial, w_ffn, norm)

            # FFN (PreNorm + residual to partial)
            ffn_out = self.ffns[i](self.ln2s[i](h))
            partial = partial + ffn_out
            V_final.append(partial)

        # ---- End of block: append to blocks lists ----
        for i in range(self.n_fields):
            blocks_list[i] = blocks_list[i] + [V_final[i]]

        return V_final, E_out, blocks_list


# =============================================================================
# MultiFieldMixer: 堆叠 AttnResMixerBlock
# =============================================================================

class MultiFieldMixer(nn.Module):
    def __init__(self, N_block, enc_dim, n_head, n_token, enc_s_dim, n_fields=2, cross_attn_heads=4):
        super().__init__()
        self.n_fields = n_fields
        self.blocks = nn.ModuleList([
            AttnResMixerBlock(
                enc_dim=enc_dim, n_head=n_head, n_token=n_token,
                enc_s_dim=enc_s_dim, n_fields=n_fields, cross_attn_heads=cross_attn_heads
            )
            for _ in range(N_block)
        ])

    def forward(self, V_list, E, edges, pos_enc):
        """
        Args:
            V_list: list of [bs, N, enc_dim] per field (from encoder)
            E: [bs, ne, enc_dim] — 共享初始边特征
            edges: [bs, ne, 2]
            pos_enc: [bs, N, enc_s_dim]

        Returns:
            V_all_list: list of [bs, N_block, N, enc_dim] per field
        """
        # 初始化 blocks_list: b_0 = encoder embedding for each field
        blocks_list = [[V_list[i]] for i in range(self.n_fields)]

        # 每个场复制一份 E 用于独立演化
        E_list = [E.clone() for _ in range(self.n_fields)]

        # 收集每个 block 的输出
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
    """每个场独立的 Decoder"""
    def __init__(self, N_block=4, enc_dim=128, enc_s_dim=10, n_fields=2):
        super().__init__()
        self.n_fields = n_fields
        # 每个场输出 1 个通道
        self.decoders = nn.ModuleList([
            Decoder(N=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, state_size=1)
            for _ in range(n_fields)
        ])

    def forward(self, V_all_list, pos_enc):
        """
        Args:
            V_all_list: list of [bs, N_block, N, enc_dim] per field
            pos_enc: [bs, N, enc_s_dim]

        Returns:
            delta: [bs, N, n_fields] — 所有场的增量拼接
        """
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
    PhysGTO-AttnRes-Multi

    融合 Block Attention Residuals + Multi-Field Cross-Attention + History Memory 的图变换器算子

    Args:
        space_size: 空间维度 (2 or 3)
        pos_enc_dim: Fourier 编码频率数
        cond_dim: 条件维度 (工艺参数)
        N_block: MixerBlock 数量
        in_dim: 输入通道数 (= n_fields)
        out_dim: 输出通道数 (= n_fields)
        enc_dim: 隐空间维度
        n_head: Attention 头数
        n_token: 投影注意力查询数
        dt: 默认时间步
        n_fields: 物理场数量 (默认 = in_dim)
        cross_attn_heads: Cross-Attention 头数
        use_memory: 是否启用 latent memory
        memory_dim: memory 维度 (默认 = enc_dim)
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
                 use_memory=True,
                 memory_dim=None,
                 ):
        super().__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.pos_enc_dim = pos_enc_dim
        self.n_fields = n_fields if n_fields is not None else in_dim
        self.use_memory = use_memory

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

        # Memory block: 每个场独立的 GRU memory
        if self.use_memory:
            mem_dim = memory_dim or enc_dim
            self.memory_blocks = nn.ModuleList([
                LatentMemoryBlock(enc_dim, mem_dim) for _ in range(self.n_fields)
            ])
            self.memory_dim = mem_dim
        else:
            self.memory_dim = 0

        self.mixer = MultiFieldMixer(
            N_block=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
            cross_attn_heads=cross_attn_heads,
        )

        self.decoder = MultiFieldDecoder(
            N_block=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
        )

    def forward_step(self, state_in, node_pos, edges, time_i, conditions,
                     memory=None, pos_enc=None, c_enc=None, dt=None):
        """
        单步预测 + memory 更新

        Args:
            state_in: [bs, N, in_dim]
            node_pos: [bs, N, space_size]
            edges: [bs, ne, 2]
            time_i: [bs,] or [bs, 1]
            conditions: [bs, cond_dim]
            memory: list of [bs, N, mem_dim] per field, or None

        Returns:
            state_pred: [bs, N, out_dim]
            memory_new: list of [bs, N, mem_dim] per field (or None)
        """
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

        # Encoder
        V_list, E = self.encoder(node_pos, state_in, t_enc, c_enc, edges_long)

        # Memory update (Encoder 后, Mixer 前)
        memory_new = None
        if self.use_memory:
            memory_new = []
            if memory is None:
                memory = [None] * self.n_fields
            for i in range(self.n_fields):
                V_list[i], mem_i = self.memory_blocks[i](V_list[i], memory[i])
                memory_new.append(mem_i)

        # Mixer
        V_all_list = self.mixer(V_list, E, edges_long, pos_enc)

        # Decoder
        v_pred = self.decoder(V_all_list, pos_enc)
        state_pred = state_in + v_pred

        return state_pred, memory_new

    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        """向后兼容: 无 memory 的单步预测"""
        state_pred, _ = self.forward_step(
            state_in, node_pos, edges, time_i, conditions,
            memory=None, pos_enc=pos_enc, c_enc=c_enc, dt=dt
        )
        return state_pred

    def autoregressive(self,
                       hist_state,
                       node_pos,
                       edges,
                       future_time_seq,
                       conditions,
                       dt=None,
                       check_point=False,
                       hist_time=None):
        """
        两阶段自回归: warm-up (历史真值) + forecast (自由 rollout)

        Args:
            hist_state: [bs, input_steps, N, C] — 历史真值序列
            node_pos: [bs, N, space_size]
            edges: [bs, ne, 2]
            future_time_seq: [bs, horizon, 1] — 未来时间序列 (相对最后历史步)
            conditions: [bs, cond_dim]
            dt: 时间步长
            check_point: 是否使用 gradient checkpointing
            hist_time: [bs, input_steps-1, 1] — 历史步间相对时间

        Returns:
            outputs: [bs, horizon, N, C] — 未来预测序列
        """
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        memory = None
        input_steps = hist_state.shape[1]

        # ---- Phase 1: Warm-up (用历史真值初始化 memory) ----
        if self.use_memory and input_steps > 1 and hist_time is not None:
            for s in range(input_steps - 1):
                state_s = hist_state[:, s]           # [bs, N, C]
                time_s = hist_time[:, s]              # [bs, 1]
                _, memory = self.forward_step(
                    state_s, node_pos, edges, time_s, conditions,
                    memory=memory, pos_enc=pos_enc, c_enc=c_enc, dt=dt
                )

        # ---- Phase 2: Forecast (自回归 rollout) ----
        state_t = hist_state[:, -1]  # 最后一个已知真值
        T = future_time_seq.shape[1]
        outputs = []

        for t in range(T):
            time_i = future_time_seq[:, t]

            def custom_forward(s_t, t_i, mem=memory):
                return self.forward_step(
                    s_t, node_pos, edges, t_i, conditions,
                    memory=mem, pos_enc=pos_enc, c_enc=c_enc, dt=dt
                )

            if check_point:
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                # checkpoint 不支持返回 tuple of list，需要包装
                state_pred, memory = checkpoint(
                    custom_forward, state_t, time_i, memory, use_reentrant=False
                )
            else:
                state_pred, memory = self.forward_step(
                    state_t, node_pos, edges, time_i, conditions,
                    memory=memory, pos_enc=pos_enc, c_enc=c_enc, dt=dt
                )

            outputs.append(state_pred)
            state_t = state_pred

        return torch.stack(outputs, dim=1)


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    print("=" * 60)
    print("PhysGTO-AttnRes-Multi + Memory 快速验证")
    print("=" * 60)

    bs, N, ne = 2, 64, 128
    T = 4
    input_steps = 3
    space_dim = 3
    in_dim = out_dim = 2
    cond_dim = 8

    model = Model(
        space_size=space_dim,
        pos_enc_dim=3,
        cond_dim=cond_dim,
        N_block=2,
        in_dim=in_dim,
        out_dim=out_dim,
        enc_dim=64,
        n_head=4,
        n_token=32,
        dt=2e-5,
        n_fields=2,
        cross_attn_heads=4,
        use_memory=True,
    )

    hist_state = torch.randn(bs, input_steps, N, in_dim)
    node_pos = torch.rand(bs, N, space_dim)
    edges = torch.randint(0, N, (bs, ne, 2))
    future_time = torch.linspace(0, 1e-4, T).unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1)
    hist_time = torch.ones(bs, input_steps - 1, 1) * 2e-5
    conditions = torch.randn(bs, cond_dim)

    # 单步 (兼容旧接口)
    pred = model(hist_state[:, -1], node_pos, edges, future_time[:, 0, :], conditions)
    print(f"[单步兼容]  pred: {pred.shape}")
    assert pred.shape == (bs, N, out_dim)

    # 单步 with memory
    pred2, mem = model.forward_step(
        hist_state[:, -1], node_pos, edges, future_time[:, 0, :], conditions
    )
    print(f"[单步+mem] pred: {pred2.shape}, memory fields: {len(mem)}, mem shape: {mem[0].shape}")
    assert pred2.shape == (bs, N, out_dim)
    assert mem[0].shape == (bs, N, 64)

    # 两阶段自回归
    out = model.autoregressive(
        hist_state, node_pos, edges, future_time, conditions,
        hist_time=hist_time
    )
    print(f"[两阶段AR] out: {out.shape}")
    assert out.shape == (bs, T, N, out_dim)

    # 参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {params/1e6:.3f}M")

    print("\nPhysGTO-AttnRes-Multi + Memory 全部验证通过!")
