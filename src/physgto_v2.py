import torch
import torch.nn as nn
import numpy as np

from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from torch.utils.checkpoint import checkpoint
from torch.amp import GradScaler, autocast

def get_edge_info(edges, node_pos):
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True) + 1e-8)
    # distance_2 = -distance_1
    E = torch.cat([d, -d, norm], dim=-1)
    return E

def FourierEmbedding(pos, pos_start, pos_length):
    # F(x) = [cos(2^i * pi * x), sin(2^i * pi * x)]
    # 高频展开，目的是拟合复杂函数而非保留原有的维度

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
    all_embeddings = torch.cat([embedding, pos], dim=-1)

    return all_embeddings

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
    def __init__(self,
                n_token=128,
                c_dim=128,
                n_heads=4,
                n_latent=4):
        super(Atten, self).__init__()

        self.c_dim = c_dim
        self.n_token = n_token
        self.n_heads = n_heads
        self.n_latent = n_latent

        # Learnable query (xavier init for better attention scale)
        self.Q = nn.Parameter(torch.empty(self.n_token, self.c_dim))
        nn.init.xavier_uniform_(self.Q)

        # W0-dependent query offset
        self.q_offset = nn.Sequential(
            nn.Linear(self.c_dim, self.c_dim),
            nn.SiLU(),
            nn.Linear(self.c_dim, self.c_dim),
        )

        # Multihead attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, dropout=0.1, batch_first=True)
        self.attention2s = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, dropout=0.1, batch_first=True) for i in range(self.n_latent)])
        self.attention3 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)

    def forward(self, W0):
        # Step 1: Initial attention with learned query + W0-dependent offset
        batch = W0.shape[0]
        learned_Q = self.Q.unsqueeze(0).expand(batch, -1, -1)
        q_bias = self.q_offset(W0.mean(dim=1, keepdim=True))  # (batch, 1, c_dim)
        learned_Q = learned_Q + q_bias
        W, _ = self.attention1(learned_Q, W0, W0)
    
        # Step 2: Self-attention on the transformed result
        for latent_atten in self.attention2s:
            W, _ = latent_atten(W, W, W)
        
        # Step 3: Position-aware attention
        W, _ = self.attention3(W0, W, W)
    
        return W

# ---------------------------
# Core modules
# ---------------------------
class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()

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
        self.f_msg_sender = MLP(input_size=edge_size, output_size=edge_size//2, n_hidden=n_hidden, act="SiLU", layer_norm=layer_norm)
        self.f_msg_receiver = MLP(input_size=edge_size, output_size=edge_size//2, n_hidden=n_hidden, act="SiLU", layer_norm=layer_norm)

        # Attention scoring networks: edge_embeddings -> scalar logit per edge
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
            n_hidden=n_hidden,
            layer_norm=layer_norm,
            act='SiLU',
            output_size=output_size
        )

    def get_edges_info(self, V, E, edges):
        # edges: (bs, ne, 2)
        # gather indices shape: (bs, ne, feat)
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        return edge_inpt

    def forward(self, V, E, edges):
        """
        V: (bs, N, node_size)
        E: (bs, ne, edge_size)
        edges: (bs, ne, 2) long
        idx0/idx1: (bs, ne) flattened indices (optional, fast path)
        """
        bs, N, _ = V.shape
        edge_inpt = self.get_edges_info(V, E, edges)
        edge_embeddings = self.f_edge(edge_inpt)

        # 显式双头消息
        msg_sender = self.f_msg_sender(edge_embeddings)
        msg_receiver = self.f_msg_receiver(edge_embeddings)

        # 注意力加权聚合
        logit_sender = self.f_attn_sender(edge_embeddings).squeeze(-1)     # (bs, ne)
        logit_receiver = self.f_attn_receiver(edge_embeddings).squeeze(-1) # (bs, ne)

        # Clamp logits to prevent overflow in scatter_softmax (exp of large values -> inf -> NaN)
        logit_sender = logit_sender.clamp(-30, 30)
        logit_receiver = logit_receiver.clamp(-30, 30)

        feat0, feat1 = msg_sender.shape[-1], msg_receiver.shape[-1]

        # IMPORTANT: expand instead of repeat (no real copy)
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

class Decoder(nn.Module):
    def __init__(self,
                 n_block = 4,
                 enc_dim=128,
                 enc_s_dim = 10,
                 state_size=1):
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

        # V_all.dim  = [bs, n_block, N, enc_dim]
        # pos_enc.dim = [bs, N, enc_s_dim]
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



class MixerBlock(nn.Module):
    def __init__(self, enc_dim, n_head, n_token, enc_s_dim):
        super().__init__()
        node_size = enc_dim + enc_s_dim

        self.gnn = GNN(
            node_size=node_size,
            edge_size=enc_dim,
            output_size=enc_dim,
            layer_norm=True
        )

        self.ln1 = nn.LayerNorm(enc_dim)
        self.ln2 = nn.LayerNorm(enc_dim)
        self.mha = Atten(n_token=n_token, c_dim=enc_dim, n_heads=n_head)

        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, 2 * enc_dim),
            nn.SiLU(),
            nn.Linear(2 * enc_dim, enc_dim)
        )

    def forward(self, V, E, edges, s_enc):
        # 1) GNN
        V_in = torch.cat([V, s_enc], dim=-1)
        v, e = self.gnn(V_in, E, edges)
        E = E + e
        V = V + v

        # 2) Attention (PreNorm)
        V = V + self.mha(self.ln1(V))

        # 3) FFN (PreNorm)
        V = V + self.ffn(self.ln2(V))

        return V, E


class Encoder(nn.Module):
    def __init__(self,
                 space_size=2,
                 state_size=4,
                 enc_dim=128,
                 enc_t_dim = 11,
                 cond_dim = 32,
                 spatial_dim = 10,
                 ):
        super(Encoder, self).__init__()

        # node embedding
        self.fv1 = MLP(input_size=state_size + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=cond_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_spatial = MLP(input_size=spatial_dim, output_size=enc_dim, act='SiLU', layer_norm=False)

        self.fuse_para = MLP(input_size=enc_dim * 3, output_size=enc_dim * 2, act='SiLU', layer_norm=False)

        # edge embedding
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)

    def forward(self, node_pos, state_in, time_i, conditions, edges, spatial_inform):
        # state_in: (bs,N,in_dim), node_pos: (bs,N,space)
        # conditions: (bs, cond_dim)  — 原始值，无需 Fourier
        # spatial_inform: (bs, 9)    — 原始值，无需 Fourier

        # node embedding
        state_in = torch.cat((state_in, node_pos), dim=-1)
        time_enc    = self.fv_time(time_i)              # (bs, enc_dim)
        cond_enc    = self.fv_cond(conditions)           # (bs, enc_dim)
        spatial_enc = self.fv_spatial(spatial_inform)    # (bs, enc_dim)

        h = torch.cat([cond_enc, time_enc, spatial_enc], dim=-1) # [B, 3*enc_dim]
        para = self.fuse_para(h)
        gamma, beta = para.chunk(2, dim=-1) # [B, enc_dim] / [B, enc_dim]

        V = self.fv1(state_in) * gamma.unsqueeze(-2) + beta.unsqueeze(-2)

        # edge embedding
        E = self.fe(get_edge_info(edges, node_pos)) # E 包含的是固定的几何信息（一直用）

        return V, E

class Mixer(nn.Module):
    def __init__(self, N, enc_dim, n_head, n_token, enc_s_dim):
        super(Mixer, self).__init__()

        self.blocks = nn.ModuleList([
            MixerBlock(enc_dim=enc_dim, n_head=n_head, n_token=n_token, enc_s_dim=enc_s_dim)
            for _ in range(N)
        ])

    def forward(self, V, E, edges_long, pos_enc):

        V_all = []
        
        for block in self.blocks:
            V, E = block(V, E, edges_long, pos_enc)
            V_all.append(V)
        
        V_all = torch.stack(V_all, dim=1) # [bs, N_block, N, enc_dim]

        return V_all

class Model(nn.Module):
    def __init__(self,
                 space_size=3,
                 pos_enc_dim=5,
                 cond_dim=32,
                 spatial_dim=10,
                 N_block=4,
                 in_dim=4,
                 out_dim=4,
                 enc_dim=128,
                 n_head=4,
                 n_token=128,
                 dt:float =0.05,
                 stepper_scheme="euler",
                 pos_x_boost=2,
                 ):
        super(Model, self).__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.pos_x_boost = pos_x_boost

        self.pos_enc_dim = pos_enc_dim
        # pos_enc: 各向异性 Fourier，输出维度与原 FourierEmbedding 相同
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        # time_enc: [t, dt] 的 Fourier(dim=pos_enc_dim) + 单独对 t 的低频 Fourier(dim=2)
        # FourierEmbedding([t, dt]) -> 2 + 2*pos_enc_dim*2 = 2 + 4*pos_enc_dim
        # FourierEmbedding_lowfreq(t)   -> 1 + 2*2 = 5
        enc_t_dim = (1 + 2 * pos_enc_dim) * 2 + (1 + 2 * 2)

        self.encoder = Encoder(
            space_size = space_size,
            state_size = in_dim,
            enc_dim = enc_dim,
            enc_t_dim = enc_t_dim,
            cond_dim = cond_dim,       # 直接接收原始条件向量
            spatial_dim = spatial_dim, # 直接接收原始空间向量（6项坐标范围+3项网格数量+1项time_ref）
            )

        self.mixer = Mixer(
            N=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim
            )

        self.decoder = Decoder(
            n_block=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            state_size=out_dim
            )

    def _encode_time(self, time_i, dt_tensor):
        """时间编码：[t,dt] 的 Fourier + t 的低频 Fourier(dim=2)"""
        time_info = torch.cat([time_i, dt_tensor], dim=-1)           # (bs, 2)
        t_fourier = FourierEmbedding(time_info, 0, self.pos_enc_dim) # (bs, 2+4*pos_enc_dim)
        t_low = FourierEmbedding(time_i, 0, 2)                       # (bs, 1+4) = (bs, 5)
        return torch.cat([t_fourier, t_low], dim=-1)                  # (bs, enc_t_dim)

    def forward(self, state_in, node_pos, edges, time_i, conditions, spatial_inform, pos_enc=None, dt=None):

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

        # Encoder（conditions 和 spatial_inform 直接传入，由 Encoder 内部 MLP 处理）
        V, E = self.encoder(node_pos, state_in, t_enc, conditions, edges_long, spatial_inform)

        # Mixer
        V_all = self.mixer(V, E, edges_long, pos_enc)

        # Decoder
        v_pred = self.decoder(V_all, pos_enc)
        if self.stepper_scheme == "euler":
            with autocast(device_type="cuda", enabled=False):
                state_pred = state_in + dt_tensor * v_pred
        elif self.stepper_scheme == "delta":
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
                       check_point=False):

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

                state_t = checkpoint(custom_forward, state_t, time_i, use_reentrant=False)

            else:
                state_t = self.forward(state_t, node_pos, edges, time_i, conditions, spatial_inform, pos_enc, dt)

            outputs.append(state_t)

        outputs = torch.stack(outputs[1:], dim=1)

        return outputs