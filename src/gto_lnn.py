"""
PhysGTO-LNN: Graph-Transformer Operator with Liquid Neural Network Components
for LPBF Physical Field Prediction

基于 PhysGTO (arXiv:2512.10227) 改进，融入 Liquid Neural Network (LNN) 组件，
专门针对 LPBF（激光粉末床熔融）熔池物理场预测的强刚性、热历史依赖特性设计。

================================================================================
改进点总览
================================================================================
1. GNN 通量方向开关 (use_directed_flux)
   - 原代码未真正区分入/出边；新增开关，True 时 e0→receiver，e1→sender
   - 符合 LPBF 热量从高温区向周围单向传导的物理规律

2. LNN-GNN 节点更新 (use_lnn_node)
   - edge_mean 作为外部驱动 u，节点特征 V 作为 LTC 隐状态 x
   - τ(V, edge_mean) 自动区分高/低导热率区域

3. LNN-FFN 替换静态 FFN (use_lnn_ffn)
   - LiquidFFN 以 V 自激励，将层间映射建模为动力学演化

4. LiquidDecoder (use_lnn_decoder)
   - 两阶段：特征压缩 + LTC 动力学输出物理增量
   - 输出热历史隐状态 h 供跨步传递

5. 自回归热历史记忆 (use_lnn_memory)
   - autoregressive() 跨步传递 h，隐式编码 LPBF 热循环历史
   - reset_memory=False 支持多段激光扫描的连续预测
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_mean


# =============================================================================
# 工具函数
# =============================================================================

def get_edge_info(edges, node_pos):
    """计算边几何特征：[位移向量, 逆位移向量, L2距离]"""
    src = torch.gather(node_pos, -2,
                       edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    dst = torch.gather(node_pos, -2,
                       edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d   = dst - src
    return torch.cat([d, -d, torch.sqrt((d ** 2).sum(-1, keepdims=True))], dim=-1)


def FourierEmbedding(pos, pos_start, pos_length):
    """正弦/余弦多频位置编码：拼接 [cos, sin, pos]"""
    shape   = pos.shape
    flat    = pos.reshape(-1, shape[-1])
    idx     = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq    = 2 ** idx * torch.pi
    cos_f   = torch.cos(freq.view(1, 1, -1) * flat.unsqueeze(-1))
    sin_f   = torch.sin(freq.view(1, 1, -1) * flat.unsqueeze(-1))
    emb     = torch.cat([cos_f, sin_f], dim=-1).view(*shape[:-1], -1)
    return torch.cat([emb, pos], dim=-1)


# =============================================================================
# 基础模块
# =============================================================================

class MLP(nn.Module):
    """多层感知机（SiLU / GELU / PReLU，可选 LayerNorm）"""

    def __init__(self, input_size=128, output_size=128, layer_norm=True,
                 n_hidden=1, hidden_size=128, act='SiLU'):
        super().__init__()
        acts   = {'GELU': nn.GELU(), 'SiLU': nn.SiLU(), 'PReLU': nn.PReLU()}
        act_fn = acts.get(act, nn.SiLU())
        if hidden_size == 0:
            layers = [nn.Linear(input_size, output_size)]
        else:
            layers = [nn.Linear(input_size, hidden_size), act_fn]
            for _ in range(1, n_hidden):
                layers += [nn.Linear(hidden_size, hidden_size), act_fn]
            layers.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                layers.append(nn.LayerNorm(output_size))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class Atten(nn.Module):
    """
    Projection-Inspired Attention（原版，保持不变）
    三步：子空间投影 → 子空间自精炼 → 重投影，O(N·M) 线性复杂度
    """

    def __init__(self, n_token=128, c_dim=128, n_heads=4):
        super().__init__()
        self.Q    = nn.Parameter(torch.randn(n_token, c_dim))
        self.attn1 = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)
        self.attn3 = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)

    def forward(self, W0):
        bs = W0.shape[0]
        Q  = self.Q.unsqueeze(0).expand(bs, -1, -1)
        W, _ = self.attn1(Q,  W0, W0)
        W, _ = self.attn2(W,  W,  W)
        W, _ = self.attn3(W0, W,  W)
        return W


# =============================================================================
# ★ Liquid Neural Network 核心组件
# =============================================================================

class LiquidCell(nn.Module):
    """
    液体时间常数单元（Liquid Time-Constant Cell, LTC）

    ODE: τ(x,u)·dx/dt = -x + gate(x,u)·f(x,u)
    离散化: x' = x + dt/τ·(-x + gate·f)

    关键特性：
    - τ 依赖状态 → 自动捕捉 LPBF 多时间尺度
      (凝固 ~μs: 小τ；固态扩散 ~ms: 大τ)
    - 输入门 sigmoid 控制外部激励注入强度
    - Tanh 有界输出 + LayerNorm → 训练稳定性
    - Heun 两阶 ODE 求解器适合凝固前沿刚性问题

    Args:
        state_dim  : 隐状态维度
        input_dim  : 外部输入维度
        dt         : 默认物理时间步
        ode_solver : 'euler' | 'heun'
        tau_min    : τ 下界（防止数值不稳定）
    """

    def __init__(self, state_dim: int, input_dim: int,
                 dt: float = 0.05, ode_solver: str = 'euler',
                 tau_min: float = 0.01):
        super().__init__()
        self.dt         = dt
        self.ode_solver = ode_solver
        self.tau_min    = tau_min
        d               = state_dim + input_dim

        self.f_net = nn.Sequential(           # 非线性映射 f(x,u)
            nn.Linear(d, state_dim * 2), nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim), nn.Tanh()
        )
        self.tau_net = nn.Sequential(         # 状态依赖时间常数 τ(x,u)
            nn.Linear(d, state_dim), nn.Softplus()
        )
        self.gate_net = nn.Sequential(        # 输入门（类 GRU）
            nn.Linear(d, state_dim), nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(state_dim)

    def _rhs(self, x, u):
        xu   = torch.cat([x, u], dim=-1)
        tau  = self.tau_net(xu).clamp(min=self.tau_min)
        gate = self.gate_net(xu)
        f    = self.f_net(xu) * gate
        return f, tau

    def _euler(self, x, u, dt):
        f, tau = self._rhs(x, u)
        return x + (1.0 / tau) * (-x + f) * dt

    def _heun(self, x, u, dt):
        """Heun（改进 Euler，2 阶），更适合刚性 ODE"""
        f1, tau1 = self._rhs(x, u)
        dx1 = (1.0 / tau1) * (-x + f1) * dt
        f2, tau2 = self._rhs(x + dx1, u)
        dx2 = (1.0 / tau2) * (-(x + dx1) + f2) * dt
        return x + 0.5 * (dx1 + dx2)

    def forward(self, x: torch.Tensor, u: torch.Tensor, dt=None) -> torch.Tensor:
        dt = dt if dt is not None else self.dt
        if self.ode_solver == 'heun':
            return self.ln(self._heun(x, u, dt))
        return self.ln(self._euler(x, u, dt))


class LiquidFFN(nn.Module):
    """
    动态前馈层：用 LTC 替代静态 FFN

    原 FFN：V' = W2·σ(W1·V)（纯静态映射）
    LiquidFFN：V' = LTC(V, V)（自激励，将层间变换视为动力学过程）

    物理意义：每层特征变换对应物理场在不同尺度下的演化；
    τ(V,V) 的自适应性使网络在高梯度区（熔池边界）
    和平缓区（固态区域）产生不同的变换速率。
    """

    def __init__(self, enc_dim: int, dt: float = 0.05, ode_solver: str = 'euler'):
        super().__init__()
        # u = V 自身（input_dim = enc_dim）
        self.liquid = LiquidCell(enc_dim, enc_dim, dt=dt, ode_solver=ode_solver)

    def forward(self, V: torch.Tensor, dt=None) -> torch.Tensor:
        return self.liquid(V, V, dt=dt)


class LiquidNodeUpdater(nn.Module):
    """
    基于 LTC 的 GNN 节点状态更新器

    原 f_node：MLP([V, edge_mean])（静态映射，忽略时序依赖）
    LiquidNodeUpdater：
        x = node_proj(V)      → LTC 隐状态（节点当前状态）
        u = edge_proj(edge_mean) → 外部物理驱动（邻域热通量）
        V' = LTC(x, u)

    LPBF 物理意义：
    - 不同节点的 τ 自适应编码局部导热率差异
      (粉末层: 大τ；致密金属: 小τ)
    - 无需显式材料属性标注，从数据中自动学习

    Args:
        node_dim  : 节点特征输出维度
        edge_dim  : edge_mean 维度（通常 = edge_size）
        input_node_dim: 节点特征输入维度（不同时需投影）
        dt        : 物理时间步
        ode_solver: ODE 求解器
    """

    def __init__(self, node_dim: int, edge_dim: int,
                 input_node_dim: int = None,
                 dt: float = 0.05, ode_solver: str = 'euler'):
        super().__init__()
        input_node_dim = input_node_dim or node_dim
        self.node_proj = (nn.Linear(input_node_dim, node_dim)
                          if input_node_dim != node_dim else nn.Identity())
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.liquid    = LiquidCell(node_dim, node_dim, dt=dt, ode_solver=ode_solver)

    def forward(self, V: torch.Tensor, edge_mean: torch.Tensor,
                dt=None) -> torch.Tensor:
        x = self.node_proj(V)
        u = self.edge_proj(edge_mean)
        return self.liquid(x, u, dt=dt)


class LiquidDecoder(nn.Module):
    """
    液体神经网络增强的 Decoder

    原 Decoder：delta_net([V_concat, pos_enc]) → δu（静态 MLP）
    LiquidDecoder 两阶段：

      阶段1 - 特征压缩：
          [V_all_concat(n_block·enc_dim), pos_enc] → MLP → enc_dim

      阶段2 - LTC 动力学输出：
          x = compressed_feat（+ h_prev 残差，注入热历史）
          u = pos_enc（空间位置条件驱动）
          V' = LTC(x, u)  →  Linear → δu

    LPBF 适配性：
    - δu 对应相变、热应力等强非线性过程，LTC 比静态 MLP 更合适
    - h_new 传递给下一时间步，隐式保留热历史
    - τ 自适应区分凝固前沿（小τ，急剧变化）与固态（大τ，缓变）

    Args:
        N_block   : GTO 块数
        enc_dim   : 隐空间维度
        enc_s_dim : 位置编码维度
        state_size: 输出物理场通道数
        dt, ode_solver: LTC 参数
    """

    def __init__(self, N_block: int = 4, enc_dim: int = 128,
                 enc_s_dim: int = 10, state_size: int = 1,
                 dt: float = 0.05, ode_solver: str = 'euler'):
        super().__init__()
        fused_dim = N_block * enc_dim + enc_s_dim

        # 阶段1：多尺度特征融合压缩
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, enc_dim * 2), nn.SiLU(),
            nn.Linear(enc_dim * 2, enc_dim), nn.LayerNorm(enc_dim)
        )

        # 阶段2：LTC 动力学（x=enc_dim, u=enc_s_dim）
        self.liquid = LiquidCell(enc_dim, enc_s_dim, dt=dt, ode_solver=ode_solver)

        # 输出头
        self.out = nn.Linear(enc_dim, state_size)

    def forward(self, V_all: torch.Tensor, pos_enc: torch.Tensor,
                hidden: torch.Tensor = None, dt=None):
        """
        Args:
            V_all  : (bs, N_block, N, enc_dim)
            pos_enc: (bs, N, enc_s_dim)
            hidden : 热历史隐状态 (bs, N, enc_dim)，None 时跳过
            dt     : 覆盖步长
        Returns:
            delta  : (bs, N, state_size) 物理场增量
            h_new  : (bs, N, enc_dim)   更新热历史隐状态
        """
        b, n_block, N, enc_dim = V_all.shape
        V_flat = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        h = self.fuse(torch.cat([V_flat, pos_enc], dim=-1))  # (bs, N, enc_dim)

        # 注入热历史（残差融合，保持梯度流）
        if hidden is not None:
            h = h + hidden

        h_new = self.liquid(h, pos_enc, dt=dt)              # (bs, N, enc_dim)
        return self.out(h_new), h_new


class StaticDecoder(nn.Module):
    """
    原版静态 Decoder（消融实验对比用）
    接口与 LiquidDecoder 对齐，返回 (delta, None)
    """

    def __init__(self, N=4, enc_dim=128, enc_s_dim=10, state_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N * enc_dim + enc_s_dim, enc_dim), nn.SiLU(),
            nn.Linear(enc_dim, enc_dim), nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc, hidden=None, dt=None):
        b, n_block, N, enc_dim = V_all.shape
        V_flat = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        return self.net(torch.cat([V_flat, pos_enc], dim=-1)), None


# =============================================================================
# ★ 改进版 GNN（通量方向开关 + LNN 节点更新）
# =============================================================================

class GNN_LNN(nn.Module):
    """
    融合 LNN 的 GNN 消息传递模块

    改进1 - 通量方向开关 (use_directed_flux):
        False（原版兼容）: e0→sender, e1→receiver（对称无向）
        True（论文原意）:  e0→receiver（入通量），e1→sender（出通量）
          物理意义：节点 i 的"入通量"=从邻居接收的热量，
                   "出通量"=向邻居散失的热量，区分热源和热汇

    改进2 - LNN 节点更新 (use_lnn_node):
        True : LiquidNodeUpdater(V, edge_mean)   # LTC 动力学更新
        False: 原版 f_node MLP（消融用）
    """

    def __init__(self, n_hidden=1, node_size=128, edge_size=128,
                 output_size=None, layer_norm=False,
                 use_directed_flux: bool = False,
                 use_lnn_node: bool = True,
                 dt: float = 0.05, ode_solver: str = 'euler'):
        super().__init__()
        self.use_directed_flux = use_directed_flux
        self.use_lnn_node      = use_lnn_node
        output_size            = output_size or node_size

        self.f_edge = MLP(edge_size + node_size * 2, edge_size,
                          n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU')

        if use_lnn_node:
            self.node_updater = LiquidNodeUpdater(
                node_dim       = output_size,
                edge_dim       = edge_size,
                input_node_dim = node_size,
                dt             = dt,
                ode_solver     = ode_solver
            )
        else:
            self.f_node = MLP(edge_size + node_size, output_size,
                              n_hidden=n_hidden, layer_norm=layer_norm, act='SiLU')

    def forward(self, V, E, edges, dt=None):
        """
        V     : (bs, N, node_size)
        E     : (bs, ne, edge_size)
        edges : (bs, ne, 2)  [...,0]=sender, [...,1]=receiver
        """
        bs, N, _ = V.shape

        # ---- 边特征更新 ----
        s = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        r = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        emb  = self.f_edge(torch.cat([s, r, E], dim=-1))   # (bs, ne, edge_size)
        e0, e1 = emb.chunk(2, dim=-1)                       # 各 edge_size//2
        h    = e0.shape[-1]

        # ---- 消息聚合 ----
        if self.use_directed_flux:
            # ★ 通量方向（论文 Flux-Oriented Message Passing 真正实现）
            idx_recv = edges[..., 1].unsqueeze(-1).expand(-1, -1, h)  # receiver
            idx_send = edges[..., 0].unsqueeze(-1).expand(-1, -1, h)  # sender
            flux_in  = scatter_mean(e0, idx_recv, dim=1, dim_size=N)   # 入通量
            flux_out = scatter_mean(e1, idx_send, dim=1, dim_size=N)   # 出通量
            edge_mean = torch.cat([flux_in, flux_out], dim=-1)
        else:
            # 原版无向等效
            i0 = edges[..., 0].unsqueeze(-1).expand(-1, -1, h)
            i1 = edges[..., 1].unsqueeze(-1).expand(-1, -1, h)
            edge_mean = torch.cat([
                scatter_mean(e0, i0, dim=1, dim_size=N),
                scatter_mean(e1, i1, dim=1, dim_size=N)
            ], dim=-1)

        # ---- 节点更新 ----
        if self.use_lnn_node:
            node_emb = self.node_updater(V, edge_mean, dt=dt)
        else:
            node_emb = self.f_node(torch.cat([V, edge_mean], dim=-1))

        return node_emb, emb


# =============================================================================
# Encoder（原版，保持不变）
# =============================================================================

class Encoder(nn.Module):
    """节点/边特征编码器（与原版 PhysGTO 完全一致）"""

    def __init__(self, space_size=2, state_size=4, enc_dim=128,
                 enc_t_dim=11, enc_c_dim=12):
        super().__init__()
        self.fv1     = MLP(state_size + space_size, enc_dim, act='SiLU', layer_norm=False)
        self.fv_time = MLP(enc_t_dim, enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(enc_c_dim, enc_dim, act='SiLU', layer_norm=False)
        self.fe      = MLP(2 * space_size + 1, enc_dim, n_hidden=1,
                           act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges):
        V = (self.fv1(torch.cat((state_in, node_pos), dim=-1))
             + self.fv_time(time_i).unsqueeze(-2)
             + self.fv_cond(conditions).unsqueeze(-2))
        E = self.fe(get_edge_info(edges, node_pos))
        return V, E


# =============================================================================
# ★ 改进版 MixerBlock
# =============================================================================

class MixerBlock_LNN(nn.Module):
    """
    融合 LNN 的 GTO 处理块

    三个子模块（均支持独立开关）：
      1) GNN_LNN   ：通量方向可控 + LNN 节点更新
      2) Atten     ：Projection-Inspired Attention（原版不变）
      3) LiquidFFN ：动态前馈（或静态 FFN，由 use_lnn_ffn 控制）
    """

    def __init__(self, enc_dim, n_head, n_token, enc_s_dim,
                 use_directed_flux=False, use_lnn_node=True,
                 use_lnn_ffn=True, dt=0.05, ode_solver='euler'):
        super().__init__()
        node_size = enc_dim + enc_s_dim

        self.gnn = GNN_LNN(
            node_size=node_size, edge_size=enc_dim, output_size=enc_dim,
            layer_norm=True, use_directed_flux=use_directed_flux,
            use_lnn_node=use_lnn_node, dt=dt, ode_solver=ode_solver
        )
        self.ln1      = nn.LayerNorm(enc_dim)
        self.ln2      = nn.LayerNorm(enc_dim)
        self.mha      = Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head)
        self._lnn_ffn = use_lnn_ffn

        if use_lnn_ffn:
            self.ffn = LiquidFFN(enc_dim, dt=dt, ode_solver=ode_solver)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(enc_dim, 2 * enc_dim), nn.SiLU(),
                nn.Linear(2 * enc_dim, enc_dim)
            )

    def forward(self, V, E, edges, s_enc, dt=None):
        # 1) GNN_LNN
        V_in = torch.cat([V, s_enc], dim=-1)
        v, e = self.gnn(V_in, E, edges, dt=dt)
        E = E + e
        V = V + v

        # 2) Attention（PreNorm + 残差）
        V = V + self.mha(self.ln1(V))

        # 3) FFN（PreNorm + 残差）
        V_n = self.ln2(V)
        V   = V + (self.ffn(V_n, dt=dt) if self._lnn_ffn else self.ffn(V_n))
        return V, E


# =============================================================================
# ★ 改进版 Mixer
# =============================================================================

class Mixer_LNN(nn.Module):
    """多块 MixerBlock_LNN 堆叠，每块输出均收集供 Decoder 多尺度融合"""

    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim,
                 use_directed_flux=False, use_lnn_node=True,
                 use_lnn_ffn=True, dt=0.05, ode_solver='euler'):
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock_LNN(
                enc_dim=enc_dim, n_head=n_head, n_token=n_token, enc_s_dim=enc_s_dim,
                use_directed_flux=use_directed_flux, use_lnn_node=use_lnn_node,
                use_lnn_ffn=use_lnn_ffn, dt=dt, ode_solver=ode_solver
            ) for _ in range(N)
        ])

    def forward(self, V, E, edges, pos_enc, dt=None):
        V_all = []
        for blk in self.blocks:
            V, E = blk(V, E, edges, pos_enc, dt=dt)
            V_all.append(V)
        return torch.stack(V_all, dim=1)   # (bs, N_block, N, enc_dim)


# =============================================================================
# ★ 完整模型：PhysGTO-LNN
# =============================================================================

class Model(nn.Module):
    """
    PhysGTO-LNN：面向 LPBF 熔池物理场预测的液体神经网络增强图变换器算子

    新增 LNN 配置项（均可独立关闭，方便消融实验）：
    ─────────────────────────────────────────────────────────────────────
    use_directed_flux (bool, 默认 False)
        启用 GNN 通量方向区分。True 时区分入/出通量，符合热传导方向性。

    use_lnn_node (bool, 默认 True)
        GNN 节点更新替换为 LiquidNodeUpdater。
        False 时退回原版静态 f_node MLP（消融用）。

    use_lnn_ffn (bool, 默认 True)
        MixerBlock FFN 替换为 LiquidFFN（自激励 LTC）。
        False 时保留原版静态 FFN（消融用）。

    use_lnn_decoder (bool, 默认 True)
        Decoder 替换为 LiquidDecoder（两阶段，含热历史接口）。
        False 时使用 StaticDecoder（原版结构，接口对齐）。

    use_lnn_memory (bool, 默认 True)
        自回归热历史记忆：autoregressive() 跨步传递 LiquidDecoder 隐状态 h。
        需要 use_lnn_decoder=True 才生效。

    ode_solver (str, 默认 'euler')
        LTC ODE 求解器。'heun' 两阶方法更适合 LPBF 刚性 ODE，计算量 ~2×。
    ─────────────────────────────────────────────────────────────────────

    热历史记忆使用方法：
        # 新样本（清空记忆）
        model.reset_hidden()
        outputs = model.autoregressive(..., reset_memory=True)

        # 多段激光扫描（保持跨段热历史）
        outputs2 = model.autoregressive(..., reset_memory=False)
    """

    def __init__(self,
                 space_size: int = 3,
                 pos_enc_dim: int = 5,
                 cond_dim: int = 32,
                 N_block: int = 4,
                 in_dim: int = 4,
                 out_dim: int = 4,
                 enc_dim: int = 128,
                 n_head: int = 4,
                 n_token: int = 128,
                 dt: float = 0.05,
                 stepper_scheme: str = 'euler',
                 # ---- LNN 配置 ----
                 use_directed_flux: bool = False,
                 use_lnn_node: bool = True,
                 use_lnn_ffn: bool = True,
                 use_lnn_decoder: bool = True,
                 use_lnn_memory: bool = True,
                 ode_solver: str = 'euler',
                 ):
        super().__init__()
        self.dt              = dt
        self.stepper_scheme  = stepper_scheme
        self.pos_enc_dim     = pos_enc_dim
        self.use_lnn_decoder = use_lnn_decoder
        self.use_lnn_memory  = use_lnn_memory and use_lnn_decoder
        self._hidden         = None   # 热历史隐状态缓冲

        # 维度计算（与原版 PhysGTO 保持一致）
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 2 * (1 + 2 * pos_enc_dim)
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.encoder = Encoder(space_size=space_size, state_size=in_dim,
                               enc_dim=enc_dim, enc_t_dim=enc_t_dim,
                               enc_c_dim=enc_c_dim)

        self.mixer = Mixer_LNN(
            N=N_block, enc_dim=enc_dim, n_head=n_head, n_token=n_token,
            enc_s_dim=enc_s_dim, use_directed_flux=use_directed_flux,
            use_lnn_node=use_lnn_node, use_lnn_ffn=use_lnn_ffn,
            dt=dt, ode_solver=ode_solver
        )

        if use_lnn_decoder:
            self.decoder = LiquidDecoder(
                N_block=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim,
                state_size=out_dim, dt=dt, ode_solver=ode_solver
            )
        else:
            self.decoder = StaticDecoder(
                N=N_block, enc_dim=enc_dim, enc_s_dim=enc_s_dim, state_size=out_dim
            )

    # ------------------------------------------------------------------
    # 热历史管理
    # ------------------------------------------------------------------

    def reset_hidden(self):
        """清空热历史隐状态，新样本/新序列推理前调用"""
        self._hidden = None

    # ------------------------------------------------------------------
    # 单步前向
    # ------------------------------------------------------------------

    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None, hidden=None):
        """
        单步预测：u_t → u_{t+1}

        Args:
            state_in  : (bs, N, in_dim)        当前物理场
            node_pos  : (bs, N, space_size)    节点空间坐标
            edges     : (bs, ne, 2)            图连接关系
            time_i    : (bs,) 或 (bs,1)        当前时刻
            conditions: (bs, cond_dim)          工艺参数（激光功率/速度等）
            pos_enc   : 预计算位置编码（可选）
            c_enc     : 预计算条件编码（可选）
            dt        : 覆盖时间步（变步长推理，如激光变速扫描）
            hidden    : 外部注入热历史 (bs, N, enc_dim)，
                        None 时使用内部缓冲

        Returns:
            state_pred : (bs, N, out_dim)  预测物理场
            h_new      : (bs, N, enc_dim)  更新后热历史隐状态（或 None）
        """
        if pos_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        if c_enc is None:
            c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        if time_i.dim() == 1:
            time_i = time_i.view(-1, 1)
        bs = time_i.shape[0]

        if dt is None:
            dt_val    = self.dt
            dt_tensor = torch.full((bs, 1), dt_val, dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (float, int)):
            dt_val    = float(dt)
            dt_tensor = torch.full((bs, 1), dt_val, dtype=time_i.dtype, device=time_i.device)
        elif isinstance(dt, (np.floating, np.integer)):  # 匹配 np.float32 等标量
            dt_val    = float(dt)
            dt_tensor = torch.tensor([dt], dtype=time_i.dtype, device=time_i.device).reshape(bs, 1)
        else:
            dt_tensor = dt.view(bs, 1).to(dtype=time_i.dtype, device=time_i.device)
            dt_val    = float(dt_tensor.mean())

        t_enc    = FourierEmbedding(torch.cat([time_i, dt_tensor], dim=-1),
                                    0, self.pos_enc_dim)
        edges_l  = edges.long() if edges.dtype != torch.long else edges

        # Encoder → Mixer → Decoder
        V, E  = self.encoder(node_pos, state_in, t_enc, c_enc, edges_l)
        V_all = self.mixer(V, E, edges_l, pos_enc, dt=dt_val)

        h_in          = hidden if hidden is not None else self._hidden
        delta, h_new  = self.decoder(V_all, pos_enc, hidden=h_in, dt=dt_val)

        return state_in + delta, h_new   # 残差预测

    # ------------------------------------------------------------------
    # 自回归推理（含热历史记忆）
    # ------------------------------------------------------------------

    def autoregressive(self,
                       state_in,
                       node_pos,
                       edges,
                       time_seq,
                       conditions,
                       dt=None,
                       check_point=False,
                       teacher_forcing=False,
                       gt_states=None,
                       reset_memory=True):
        """
        自回归多步预测，核心循环传递 LNN 热历史隐状态

        热历史机制：
            h_0 = None（或上一段的 h_T）
            for t in 0..T-1:
                u_{t+1}, h_{t+1} = forward(u_t, ..., hidden=h_t)
            # h_T 保存在 self._hidden，供后续段使用

        Args:
            state_in     : (bs, N, in_dim)       初始物理场
            node_pos     : (bs, N, space_size)
            edges        : (bs, ne, 2)
            time_seq     : (bs, T)               时间序列
            conditions   : (bs, cond_dim)         工艺参数
            dt           : 覆盖步长
            teacher_forcing: 训练时使用 GT 状态作为下步输入
            gt_states    : (bs, T, N, in_dim)
            reset_memory : True  → 清空热历史（新样本）
                           False → 保持热历史（多段连续扫描）

        Returns:
            outputs : (bs, T, N, out_dim)  预测序列（不含初始帧）
        """
        if reset_memory:
            self.reset_hidden()

        state_t = state_in
        outputs = []
        hidden  = self._hidden
        T       = time_seq.shape[1]

        # 预计算固定编码（节省循环内重复计算）
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        for t in range(T):
            state_pred, hidden = self.forward(
                state_in=state_t, node_pos=node_pos, edges=edges,
                time_i=time_seq[:, t], conditions=conditions,
                pos_enc=pos_enc, c_enc=c_enc, dt=dt,
                hidden=hidden if self.use_lnn_memory else None
            )
            outputs.append(state_pred)

            if t < T - 1:
                state_t = (gt_states[:, t]
                           if teacher_forcing and gt_states is not None
                           else state_pred)

        # 保存最终热历史（detach 防止梯度跨序列传播，但保留值）
        if hidden is not None:
            self._hidden = hidden.detach()

        return torch.stack(outputs, dim=1)   # (bs, T, N, out_dim)


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    print("=" * 60)
    print("PhysGTO-LNN 快速验证（LPBF 配置）")
    print("=" * 60)

    bs, N, ne = 2, 256, 512
    T         = 8
    space_dim, in_dim, out_dim, cond_dim = 3, 4, 4, 8

    def build(flux=False, lnn_node=True, lnn_ffn=True,
              lnn_dec=True, lnn_mem=True, solver='euler'):
        return Model(
            space_size=space_dim, pos_enc_dim=3, cond_dim=cond_dim,
            N_block=2, in_dim=in_dim, out_dim=out_dim,
            enc_dim=64, n_head=4, n_token=32,
            dt=1e-5, use_directed_flux=flux,
            use_lnn_node=lnn_node, use_lnn_ffn=lnn_ffn,
            use_lnn_decoder=lnn_dec, use_lnn_memory=lnn_mem,
            ode_solver=solver
        )

    state_in   = torch.randn(bs, N, in_dim)
    node_pos   = torch.rand(bs, N, space_dim)
    edges      = torch.randint(0, N, (bs, ne, 2))
    time_seq   = torch.linspace(0, 1e-4, T).unsqueeze(0).expand(bs, -1)
    conditions = torch.randn(bs, cond_dim)

    # ---- 全 LNN + Heun ----
    m = build(flux=True, solver='heun')
    m.reset_hidden()
    pred, h = m(state_in, node_pos, edges, time_seq[:, 0], conditions)
    print(f"[单步-Heun]   pred:{pred.shape}  h:{h.shape}")
    assert pred.shape == (bs, N, out_dim) and h.shape == (bs, N, 64)

    m.reset_hidden()
    out = m.autoregressive(state_in, node_pos, edges, time_seq,
                            conditions, reset_memory=True)
    print(f"[自回归]      out:{out.shape}")
    assert out.shape == (bs, T, N, out_dim)

    # 连续扫描（reset_memory=False 保持热历史）
    out2 = m.autoregressive(out[:, -1], node_pos, edges, time_seq,
                              conditions, reset_memory=False)
    print(f"[连续段]      out2:{out2.shape}")

    # ---- 消融：全关 LNN（等效原版）----
    m_abl = build(flux=False, lnn_node=False, lnn_ffn=False,
                  lnn_dec=False, lnn_mem=False)
    pred_abl, _ = m_abl(state_in, node_pos, edges, time_seq[:, 0], conditions)
    print(f"[消融-原版]   pred:{pred_abl.shape}")

    # ---- 参数量对比 ----
    p_lnn = sum(p.numel() for p in m.parameters())
    p_abl = sum(p.numel() for p in m_abl.parameters())
    print(f"\n参数量  PhysGTO-LNN: {p_lnn/1e6:.3f}M  |  静态基线: {p_abl/1e6:.3f}M")
    print(f"参数开销: +{(p_lnn - p_abl)/p_abl*100:.1f}%")

    print("\n✅  PhysGTO-LNN 全部验证通过！")