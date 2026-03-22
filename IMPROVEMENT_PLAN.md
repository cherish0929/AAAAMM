# PhysGTO 速度场预测改进方案

> 针对 `src/physgto.py` 和 `src/physgto_res.py` 的架构改进计划
> 目标：将速度场 L2 误差从 ~0.6 进一步降低，不嵌入过多物理信息，通过算子学习方式改善效果

---

## 一、现状分析

### 1.1 当前架构概览

```
Encoder → Mixer (N_block × MixerBlock) → Decoder → delta_pred → state_pred = state_in + delta
```

- **Encoder**: 节点特征 = state + pos (+ vel_mag in physgto.py); 边特征 = 空间距离信息 (d, -d, norm), 共7维
- **MixerBlock**: GNN (scatter_mean聚合) → Attention (Perceiver式 learned query) → FFN, 各有PreNorm
- **Decoder**: 拼接所有block输出 → Linear → GatedResBlock → Linear → delta
- **Autoregressive**: 单步预测 + 循环推进，支持 teacher forcing

### 1.2 已实施的改进（第一轮，效果从 L2~1.0 降到 ~0.6）

1. **速度幅值辅助输入** (`physgto.py`): Encoder 额外拼接 `||U||` 作为节点特征
2. **Decoder 门控残差**: proj → res_block → sigmoid gate → gated residual → output
3. **MixerBlock PreNorm + 可学习残差缩放**: `alpha_gnn, alpha_attn, alpha_ffn` 参数
4. **直接增量预测**: `state_pred = state_in + delta` (无 dt 缩放)

### 1.3 核心瓶颈诊断

| 瓶颈 | 严重程度 | 说明 |
|------|---------|------|
| **GNN scatter_mean 各向同性平均** | ★★★★★ | 熔池边界存在尖锐速度梯度，scatter_mean 将所有邻居等权平均，抹杀了剪切层和速度不连续面的信息。这是速度场预测最关键的瓶颈 |
| **边特征缺乏速度梯度信息** | ★★★★ | 当前边特征仅包含空间距离 (d, -d, norm)，完全不包含沿边的速度差异。GNN 无法在消息传递中感知速度梯度的方向和强度 |
| **单一 Decoder 头处理 Ux/Uy/Uz** | ★★★★ | 三个速度分量在 LPBF 中物理特性差异巨大：Uy 受重力/浮力驱动（竖直），Ux/Uz 受 Marangoni 对流驱动（水平）。用同一个线性头预测三者限制了模型对各分量不同动力学的适应能力 |
| **自回归误差累积** | ★★★★ | 训练时模型只见到"完美输入"（GT 或刚预测的单步），20 步推断中误差单调累积，无自纠错能力 |
| **无空间自适应残差缩放** | ★★★ | delta 在所有节点等权施加，但 LPBF 中熔池需要大幅更新，背景区域应接近零更新 |

---

## 二、改进方案详解

### 2.1 【高优先级】GNN 边门控注意力聚合（替换 scatter_mean）

**问题**: `scatter_mean` 对每个节点的所有邻居消息取平均，这在速度场中是灾难性的——熔池边界处一侧速度极大、另一侧接近零，平均后信息完全丢失。

**方案**: 引入 `GatedGNN`，为每条边学习一个注意力权重，通过 `scatter_softmax` 实现加权聚合。

```python
class GatedGNN(nn.Module):
    """
    边门控注意力GNN：替换 scatter_mean 为 scatter_softmax + scatter_add
    让网络学习每条边的重要性权重，保留尖锐梯度信息
    """
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super().__init__()
        output_size = output_size or node_size

        # 边更新：sender + receiver + edge_feat -> 新边嵌入
        self.f_edge = MLP(
            input_size=edge_size + node_size * 2,
            n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU',
            output_size=edge_size
        )

        # 边注意力打分网络：学习每条边的聚合权重
        self.f_attn = nn.Sequential(
            nn.Linear(edge_size, edge_size // 2),
            nn.SiLU(),
            nn.Linear(edge_size // 2, 1),
        )

        # 节点更新
        self.f_node = MLP(
            input_size=edge_size + node_size,
            n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU',
            output_size=output_size
        )

    def forward(self, V, E, edges):
        bs, N, _ = V.shape
        ne = edges.shape[1]

        # Gather sender/receiver
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        # 分成两个方向的消息
        edge_emb_0, edge_emb_1 = edge_embeddings.chunk(2, dim=-1)
        feat_dim = edge_emb_0.shape[-1]

        # 计算注意力logits
        attn_logits_0 = self.f_attn(edge_emb_0).squeeze(-1)  # [bs, ne]
        attn_logits_1 = self.f_attn(edge_emb_1).squeeze(-1)

        recv_idx_0 = edges[..., 0]  # 消息聚合到 sender 节点
        recv_idx_1 = edges[..., 1]  # 消息聚合到 receiver 节点

        # scatter_softmax: 对每个节点的所有入边做 softmax
        attn_weights_0 = scatter_softmax(attn_logits_0, recv_idx_0, dim=1, dim_size=N)
        attn_weights_1 = scatter_softmax(attn_logits_1, recv_idx_1, dim=1, dim_size=N)

        # 加权聚合 (scatter_add 而非 scatter_mean)
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
```

**关键依赖**: 需要 `from torch_scatter import scatter_softmax`（torch_scatter 已安装，scatter_softmax 包含在内）

**为什么有效**:
- 在熔池边界，模型可以学到"高速侧邻居权重大、低速侧权重小"
- 在背景区域，所有邻居权重接近均匀（退化为 scatter_mean）
- 不增加计算量（f_attn 很小），但显著增加表达能力

---

### 2.2 【高优先级】边特征注入速度差异（Encoder 改进）

**问题**: 当前边嵌入 `E = fe(d, -d, norm)` 仅包含空间几何信息，GNN 的消息传递完全不知道沿每条边速度是如何变化的。

**方案**: 在 Encoder 中计算每条边的速度差异 `vel_diff = recv_vel - send_vel`，拼接到边特征中。

```python
class Encoder(nn.Module):
    def __init__(self, space_size, state_size, enc_dim, enc_t_dim, enc_c_dim):
        super().__init__()
        # +1 for velocity magnitude
        self.fv1 = MLP(input_size=state_size + space_size + 1, output_size=enc_dim, ...)
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, ...)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim, ...)

        # 边特征：空间信息 (7维) + 速度差异 (state_size维)
        self.fe = MLP(
            input_size=2 * space_size + 1 + state_size,  # 原来是 2*space_size+1 = 7
            output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False
        )

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        # 节点嵌入（同现有逻辑，加 vel_mag）
        vel_mag = torch.norm(state_in, dim=-1, keepdim=True)
        state_aug = torch.cat((state_in, node_pos, vel_mag), dim=-1)
        time_enc = self.fv_time(time_i)
        cond_enc = self.fv_cond(conditions)
        V = self.fv1(state_aug) + time_enc.unsqueeze(-2) + cond_enc.unsqueeze(-2)

        # 边嵌入：空间信息 + 速度差异
        spatial_edge = get_edge_info(edges, node_pos)  # [bs, ne, 7]

        send_vel = torch.gather(state_in, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, state_in.shape[-1]))
        recv_vel = torch.gather(state_in, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, state_in.shape[-1]))
        vel_diff = recv_vel - send_vel  # [bs, ne, state_size]

        edge_input = torch.cat([spatial_edge, vel_diff], dim=-1)  # [bs, ne, 7 + state_size]
        E = self.fe(edge_input)

        return V, E
```

**为什么有效**:
- 速度差异直接告诉 GNN "这条边跨越了多大的速度梯度"
- 在熔池-固体界面处 vel_diff 很大 → GNN 自然关注这些边
- 在均匀区域 vel_diff ≈ 0 → 不干扰正常消息传递
- 只增加 state_size (=3) 维的输入，开销极小

---

### 2.3 【高优先级】分量独立 Decoder 头 + 空间自适应残差缩放

**问题**:
1. Ux/Uy/Uz 用同一个线性层输出，但三个分量物理性质差异大
2. delta 在所有节点等权施加，但背景区域应该几乎不更新

**方案**: 共享 backbone + 独立输出头 + 空间自适应门控

```python
class Decoder(nn.Module):
    def __init__(self, N=4, enc_dim=128, enc_s_dim=10, state_size=3):
        super().__init__()
        self.state_size = state_size
        in_dim = N * enc_dim + enc_s_dim

        # 共享 backbone（提取共性特征）
        self.proj = nn.Linear(in_dim, enc_dim)
        self.res_block = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, enc_dim * 2),
            nn.SiLU(),
            nn.Linear(enc_dim * 2, enc_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.Sigmoid(),
        )
        self.backbone_norm = nn.LayerNorm(enc_dim)

        # 每个速度分量独立的输出头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_dim, enc_dim // 2),
                nn.SiLU(),
                nn.Linear(enc_dim // 2, 1),
            )
            for _ in range(state_size)
        ])

        # 空间自适应残差门：学习每个节点的 delta 应该放大还是缩小
        # 熔池区域需要大更新，背景区域需要近零更新
        self.spatial_gate = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.SiLU(),
            nn.Linear(enc_dim // 2, state_size),
            nn.Tanh(),  # 输出 [-1, 1]，作为缩放因子
        )

    def forward(self, V_all, pos_enc):
        b, n_block, N, enc_dim = V_all.shape
        V_all = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        h = self.proj(torch.cat([V_all, pos_enc], dim=-1))
        r = self.res_block(h)
        g = self.gate(h)
        h = h + g * r
        h = self.backbone_norm(h)

        # 分量独立预测
        components = [head(h) for head in self.heads]   # list of [b, N, 1]
        raw_delta = torch.cat(components, dim=-1)        # [b, N, state_size]

        # 空间自适应缩放
        scale = self.spatial_gate(h)                     # [b, N, state_size]，范围 [-1, 1]
        delta = raw_delta * (1.0 + scale)                # 调制每个节点的更新幅度

        return delta
```

**为什么有效**:
- **分量独立头**: Uy 的头可以学到重力/浮力主导的模式，Ux/Uz 的头学到 Marangoni 对流模式
- **空间自适应门**: `scale ≈ -1` 时 delta → 0（背景区域），`scale ≈ 1` 时 delta 放大2倍（活跃区域）
- backbone 共享避免参数爆炸

---

### 2.4 【高优先级】Pushforward Trick（自回归误差纠正）

**问题**: 训练时模型输入是完美的 GT 状态，但推断时用的是自己之前的预测（含误差）。随着 20 步推进，小误差被放大为大偏差。

**方案**: 训练期间在自回归输入中注入小噪声，强迫模型学会从"不完美的输入"中恢复。这就是所谓的 **Pushforward Trick**（Brandstetter et al., 2022, "Message Passing Neural PDE Solvers"）。

```python
def autoregressive(self, state_in, node_pos, edges, time_seq, conditions,
                   dt=None, teacher_forcing=False, gt_states=None):
    state_t = state_in
    outputs = [state_in]
    pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
    c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

    T = time_seq.shape[1]
    for t in range(T):
        time_i = time_seq[:, t]

        # Pushforward trick: 训练时对非首步输入加小噪声
        if self.training and t > 0:
            noise_scale = 0.02  # 归一化空间中的小噪声
            noise = torch.randn_like(state_t) * noise_scale
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
```

**参考文献**:
- Brandstetter, J., Welling, M., & Worrall, D. (2022). "Message Passing Neural PDE Solvers." *ICLR 2022*.
- Sanchez-Gonzalez et al. (2020). "Learning to Simulate Complex Physics with Graph Networks." *ICML 2020*.

**为什么有效**:
- 噪声模拟了自回归推断中的预测误差分布
- 模型被迫学会"纠错"而非仅从完美输入预测
- noise_scale = 0.02 足够小不影响学习，足够大可以训练鲁棒性
- 仅训练时生效，推断时无额外开销

---

## 三、两个模型文件的具体修改清单

### 3.1 `src/physgto.py` 修改清单

| 序号 | 位置 | 修改内容 | 影响 |
|------|------|---------|------|
| 1 | import | 添加 `from torch_scatter import scatter_softmax` | 新依赖（已安装） |
| 2 | `GNN` 类 | **替换为 `GatedGNN` 类**（见2.1节代码） | 核心改进 |
| 3 | `MixerBlock.__init__` | `self.gnn = GatedGNN(...)` 替换 `self.gnn = GNN(...)` | 使用新GNN |
| 4 | `Encoder.__init__` | `self.fe` 输入维度从 `2*space_size+1` 改为 `2*space_size+1+state_size` | 边特征加速度差异 |
| 5 | `Encoder.forward` | 添加 vel_diff 计算和拼接逻辑（见2.2节代码） | 边特征增强 |
| 6 | `Decoder` 类 | **重写**：添加分量独立头 + 空间自适应门（见2.3节代码） | Decoder增强 |
| 7 | `Model.autoregressive` | 添加 pushforward trick 噪声注入（见2.4节代码） | 自回归鲁棒性 |

### 3.2 `src/physgto_res.py` 修改清单

与 `physgto.py` 完全相同的修改，额外注意：

| 序号 | 额外注意事项 |
|------|-------------|
| 1 | `physgto_res.py` 的 `Encoder` 没有 vel_mag（`input_size=state_size+space_size`），需同步添加 `+1` |
| 2 | `physgto_res.py` 的 `MixerBlock` 没有 `ln0` 和 `alpha_*` 参数，需同步添加 PreNorm 和可学习缩放 |
| 3 | `physgto_res.py` 的 `Model.forward` 有 `dt_tensor` 和 `time_info = cat([time_i, dt_tensor])` 逻辑，需保留 |
| 4 | `physgto_res.py` 的 `broadcast_dt` 函数需保留 |

---

## 四、完整修改后的代码模板

### 4.1 `physgto.py` 完整修改后代码

```python
import torch
import torch.nn as nn
import numpy as np

from torch_scatter import scatter_mean, scatter_add, scatter_softmax

VELOCITY_FIELD_NAMES = ("Ux", "Uy", "Uz")

def get_edge_info(edges, node_pos):
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True))
    E = torch.cat([d, -d, norm], dim=-1)
    return E

class MLP(nn.Module):
    def __init__(self,
                input_size = 128,
                output_size = 128,
                layer_norm = True,
                n_hidden=1,
                hidden_size = 128,
                act = 'SiLU',
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
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)

class Atten(nn.Module):
    def __init__(self,
                n_token=128,
                c_dim=128,
                n_heads=4):
        super(Atten, self).__init__()

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
    GNN with edge-gated attention aggregation.
    替换 scatter_mean 为 scatter_softmax + scatter_add，
    让网络学习每条边的聚合权重，保留尖锐速度梯度信息。
    """
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

        self.f_attn = nn.Sequential(
            nn.Linear(edge_size, edge_size // 2),
            nn.SiLU(),
            nn.Linear(edge_size // 2, 1),
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

        edge_emb_0, edge_emb_1 = edge_embeddings.chunk(2, dim=-1)
        feat_dim = edge_emb_0.shape[-1]

        attn_logits_0 = self.f_attn(edge_emb_0).squeeze(-1)
        attn_logits_1 = self.f_attn(edge_emb_1).squeeze(-1)

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


class Decoder(nn.Module):
    """
    分量独立 Decoder：共享backbone + 独立Ux/Uy/Uz输出头 + 空间自适应残差门
    """
    def __init__(self, N=4, enc_dim=128, enc_s_dim=10, state_size=3):
        super().__init__()
        self.state_size = state_size
        in_dim = N * enc_dim + enc_s_dim

        self.proj = nn.Linear(in_dim, enc_dim)
        self.res_block = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, enc_dim * 2),
            nn.SiLU(),
            nn.Linear(enc_dim * 2, enc_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.Sigmoid(),
        )
        self.backbone_norm = nn.LayerNorm(enc_dim)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_dim, enc_dim // 2),
                nn.SiLU(),
                nn.Linear(enc_dim // 2, 1),
            )
            for _ in range(state_size)
        ])

        self.spatial_gate = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.SiLU(),
            nn.Linear(enc_dim // 2, state_size),
            nn.Tanh(),
        )

    def forward(self, V_all, pos_enc):
        b, n_block, N, enc_dim = V_all.shape
        V_all = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        h = self.proj(torch.cat([V_all, pos_enc], dim=-1))
        r = self.res_block(h)
        g = self.gate(h)
        h = h + g * r
        h = self.backbone_norm(h)

        components = [head(h) for head in self.heads]
        raw_delta = torch.cat(components, dim=-1)

        scale = self.spatial_gate(h)
        delta = raw_delta * (1.0 + scale)

        return delta


class MixerBlock(nn.Module):
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        node_size = enc_dim + enc_s_dim

        self.gnn = GatedGNN(   # <-- 改为 GatedGNN
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
    def __init__(self, space_size=2, state_size=4, enc_dim=128, enc_t_dim=11, enc_c_dim=12):
        super().__init__()

        self.fv1 = MLP(input_size=state_size + space_size + 1, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim, act='SiLU', layer_norm=False)

        # 边特征：空间信息 (7维) + 速度差异 (state_size维)
        self.fe = MLP(input_size=2 * space_size + 1 + state_size, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        vel_mag = torch.norm(state_in, dim=-1, keepdim=True)
        state_aug = torch.cat((state_in, node_pos, vel_mag), dim=-1)
        time_enc = self.fv_time(time_i)
        cond_enc = self.fv_cond(conditions)
        V = self.fv1(state_aug) + time_enc.unsqueeze(-2) + cond_enc.unsqueeze(-2)

        # 空间边信息
        spatial_edge = get_edge_info(edges, node_pos)  # [bs, ne, 7]

        # 沿边的速度差异
        send_vel = torch.gather(state_in, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, state_in.shape[-1]))
        recv_vel = torch.gather(state_in, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, state_in.shape[-1]))
        vel_diff = recv_vel - send_vel  # [bs, ne, state_size]

        edge_input = torch.cat([spatial_edge, vel_diff], dim=-1)
        E = self.fe(edge_input)

        return V, E


class Mixer(nn.Module):
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock(enc_dim=enc_dim, n_head=n_head, n_token=n_token, enc_s_dim=enc_s_dim)
            for _ in range(N)
        ])

    def forward(self, V, E, edges_long, pos_enc):
        V_all = []
        for block in self.blocks:
            V, E = block(V, E, edges_long, pos_enc)
            V_all.append(V)
        V_all = torch.stack(V_all, dim=1)
        return V_all


class Model(nn.Module):
    def __init__(self, space_size=3, pos_enc_dim=5, cond_dim=32, N_block=4,
                 in_dim=4, out_dim=4, enc_dim=128, n_head=4, n_token=128,
                 dt:float=0.05, stepper_scheme="euler"):
        super().__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.out_dim = out_dim
        self.pos_enc_dim = pos_enc_dim
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 1 + 2 * pos_enc_dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.encoder = Encoder(
            space_size=space_size, state_size=in_dim, enc_dim=enc_dim,
            enc_t_dim=enc_t_dim, enc_c_dim=enc_c_dim
        )
        self.mixer = Mixer(N=N_block, enc_dim=enc_dim, n_head=n_head, n_token=n_token, enc_s_dim=enc_s_dim)
        self.decoder = Decoder(N=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, state_size=out_dim)

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

        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim)
        edges_long = edges.long() if edges.dtype != torch.long else edges
        V, E = self.encoder(node_pos, state_in, t_enc, c_enc, edges_long)
        V_all = self.mixer(V, E, edges_long, pos_enc)
        delta_pred = self.decoder(V_all, pos_enc)
        state_pred = state_in + delta_pred
        return state_pred

    def autoregressive(self, state_in, node_pos, edges, time_seq, conditions,
                       dt=None, teacher_forcing=False, gt_states=None):
        state_t = state_in
        outputs = [state_in]
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        T = time_seq.shape[1]
        for t in range(T):
            time_i = time_seq[:, t]

            # ===== Pushforward Trick =====
            if self.training and t > 0:
                noise_scale = 0.02
                noise = torch.randn_like(state_t) * noise_scale
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
```

### 4.2 `physgto_res.py` 完整修改后代码

与 physgto.py 相同的改进，但需保留其特有的 `broadcast_dt` 函数和 `time_info = cat([time_i, dt_tensor])` 逻辑。具体差异：

```python
# physgto_res.py 的 Encoder 需要保留 broadcast_dt 和 dt 相关逻辑
# enc_t_dim 是 2*(1+2*pos_enc_dim) 而非 1+2*pos_enc_dim

# Model.forward 中保留：
#   dt_tensor 的处理逻辑
#   time_info = torch.cat([time_i, dt_tensor], dim=-1)
#   t_enc = FourierEmbedding(time_info, 0, self.pos_enc_dim)

# 其余改动（GatedGNN, Encoder边特征, Decoder分量头, Pushforward）完全一致
```

---

## 五、可选的进一步改进（中等优先级）

以下改进在四项核心改进验证有效后可以继续尝试：

### 5.1 时间步感知的噪声缩放

```python
# 在 autoregressive 中，噪声随时间步增大（模拟真实的误差累积模式）
if self.training and t > 0:
    noise_scale = 0.01 + 0.01 * (t / T)  # 从0.01线性增到0.02
    noise = torch.randn_like(state_t) * noise_scale
    state_t_input = state_t + noise
```

### 5.2 时序一致性正则化

在训练损失中添加连续步之间 delta 的平滑约束：

```python
# 在 train.py 的损失计算中添加
if outputs.shape[1] > 1:
    delta_t = outputs[:, 1:] - outputs[:, :-1]      # 连续步的差异
    delta_gt = gt_states[:, 1:] - gt_states[:, :-1]
    smooth_loss = F.smooth_l1_loss(delta_t, delta_gt, beta=0.05)
    total_loss += 0.05 * smooth_loss
```

### 5.3 Scheduled Sampling 改进

当前 teacher forcing 是线性衰减，可以改为更激进的二次衰减：

```python
# 更激进的 teacher forcing 衰减
if epoch < decay_start:
    teacher_prob = 1.0
elif epoch < decay_end:
    progress = (epoch - decay_start) / (decay_end - decay_start)
    teacher_prob = (1.0 - progress) ** 2  # 二次衰减，更快地让模型独立
else:
    teacher_prob = 0.0
```

### 5.4 多分辨率边子采样

在不同的 MixerBlock 中使用不同比例的边子采样，让浅层block看全局、深层block看局部：

```python
# 概念性代码（需修改 Mixer 的 forward）
for i, block in enumerate(self.blocks):
    if i < len(self.blocks) // 2:
        # 浅层：使用较少的边（全局粗粒度）
        edge_mask = random_subsample(edges, ratio=0.3)
    else:
        # 深层：使用更多的边（局部精细）
        edge_mask = random_subsample(edges, ratio=1.0)
    V, E = block(V, E[edge_mask], edges[edge_mask], pos_enc)
```

---

## 六、理论依据与参考文献

### 6.1 为什么 scatter_mean 对速度场不好

在 CFD 中，速度场的一个核心特征是**不连续性和尖锐梯度**（如激波、剪切层、相界面）。scatter_mean 本质上是一个**低通滤波器**——它将邻域信息平均化，相当于空间平滑。这对于温度场（扩散主导、平滑变化）是合理的，但对速度场会：

1. 模糊熔池-固体界面的速度跳变
2. 抹杀 Marangoni 涡流的细节
3. 让模型无法区分"所有邻居速度相同=0"和"一半邻居+1、一半邻居-1"

边门控注意力让模型自适应地选择聚合策略，在平滑区域退化为均值，在梯度区域实现方向选择性聚合。

### 6.2 Pushforward Trick 的数学直觉

设 $f_\theta$ 是神经网络算子，$\hat{u}^t$ 是第 $t$ 步预测，$u^t$ 是真值。标准训练只优化：

$$\mathcal{L} = \sum_t \|f_\theta(u^{t-1}) - u^t\|^2$$

但推断时的输入是 $\hat{u}^{t-1}$ 而非 $u^{t-1}$，存在**分布偏移**（distribution shift）。Pushforward trick 通过在训练时使用：

$$\mathcal{L}_{pf} = \sum_t \|f_\theta(u^{t-1} + \epsilon^{t-1}) - u^t\|^2, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

使模型在"有噪声的输入"上训练，从而在推断时对自身预测误差有更好的鲁棒性。

### 6.3 参考文献

1. **Brandstetter, J., Welling, M., & Worrall, D.** (2022). "Message Passing Neural PDE Solvers." *ICLR 2022*. — Pushforward trick 的原始提出
2. **Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P.** (2021). "Learning Mesh-Based Simulation with Graph Networks." *ICLR 2021*. — 图网络用于 CFD 模拟的基础工作
3. **Sanchez-Gonzalez, A., et al.** (2020). "Learning to Simulate Complex Physics with Graph Networks." *ICML 2020*. — GNN 物理模拟的开创性工作
4. **Li, Z., et al.** (2020). "Fourier Neural Operator for Parametric Partial Differential Equations." *NeurIPS 2020*. — FNO，神经算子学习的基础
5. **Veličković, P., et al.** (2018). "Graph Attention Networks." *ICLR 2018*. — GAT，边注意力机制的基础
6. **Corso, G., et al.** (2020). "Principal Neighbourhood Aggregation for Graph Nets." *NeurIPS 2020*. — 多种聚合方式的组合优于单一 mean/sum
7. **Lam, R., et al.** (2023). "GraphCast: Learning skillful medium-range global weather forecasting." *Science*. — 大规模 GNN 用于天气预测（流体物理场），使用了类似的编码器增强策略
8. **Bonnet, F., et al.** (2022). "AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions." *NeurIPS 2022 Datasets*. — CFD 速度场的 GNN benchmark

---

## 七、预期效果与风险评估

### 7.1 预期效果

| 改进 | 预期 L2 误差下降 | 信心 |
|------|-----------------|------|
| GatedGNN (替换 scatter_mean) | 10-20% | 高 |
| 边特征速度差异 | 5-15% | 高 |
| 分量独立 Decoder + 空间门 | 5-10% | 中高 |
| Pushforward Trick | 长程推断改善 15-30% | 高 |
| **综合** | **L2 从 ~0.6 降到 ~0.35-0.45** | 中 |

### 7.2 风险与注意事项

1. **scatter_softmax 数值稳定性**: 如果某个节点只有1条边，softmax 退化为常数1.0，无影响；但如果 attn_logits 过大可能导致 NaN。建议在 f_attn 中初始化最后一层的 bias 为 0。

2. **Pushforward 噪声量**: noise_scale=0.02 是经验值。如果归一化后的速度范围不是 [-1,1] 左右，需要根据实际数据范围调整。建议先用 0.01 试跑看效果。

3. **分量独立头增加参数量**: 3 个独立头 vs 1 个统一头增加约 `3 * (enc_dim * enc_dim/2 + enc_dim/2)` ≈ 25K 参数，在总参数量面前可忽略。

4. **内存开销**: GatedGNN 的 scatter_softmax 需要额外存储 attn_weights，内存增加约 5-10%。对于 batch_size=1 的情况影响很小。

5. **训练时间**: 主要开销来自 GatedGNN 中的额外 f_attn 前向传播和 scatter_softmax，预计训练时间增加 10-15%。

---

## 八、实施步骤（推荐顺序）

1. **第一步**: 修改 `physgto.py` 的 GNN → GatedGNN + Encoder 边特征增强
2. **第二步**: 修改 `physgto.py` 的 Decoder → 分量独立头 + 空间门
3. **第三步**: 修改 `physgto.py` 的 autoregressive → Pushforward Trick
4. **第四步**: 训练验证 physgto.py 的改进效果
5. **第五步**: 将同样改动应用到 `physgto_res.py`（注意保留其 dt 处理逻辑）
6. **第六步**: 对比两个模型的效果，选择更优者继续迭代

每一步修改后建议先做小规模验证（跑 20-50 个 epoch 看趋势），确认无 bug 且趋势正确后再做完整训练。
