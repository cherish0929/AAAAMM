import torch
import torch.nn as nn
import numpy as np

from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint
from torch.amp import GradScaler, autocast

def get_edge_info(edges, node_pos):
    senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).expand(-1, -1, node_pos.shape[-1]))
    d = receivers - senders
    norm = torch.sqrt((d ** 2).sum(-1, keepdims=True))
    # distance_2 = -distance_1
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
        
        # Learnable query
        self.Q = nn.Parameter(torch.randn(self.n_token, self.c_dim), requires_grad=True)

        # Multihead attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=self.c_dim, num_heads=self.n_heads, batch_first=True)

    def forward(self, W0):   
        # Step 1: Initial attention with learned query
        batch = W0.shape[0]
        learned_Q = self.Q.unsqueeze(0).repeat(batch, 1, 1)
        W, _ = self.attention1(learned_Q, W0, W0)
    
        # Step 2: Self-attention on the transformed result
        W, _ = self.attention2(W, W, W)
        
        # Step 3: Position-aware attention
        W, _ = self.attention3(W0, W, W)
    
        return W
    
def FourierEmbedding(pos, pos_start, pos_length):
    # F(x) = [cos(2^i * pi * x), sin(2^i * pi * x)]
    
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

        # keep your original semantics: split into two directed parts
        edge_embeddings_0, edge_embeddings_1 = edge_embeddings.chunk(2, dim=-1)

        feat0 = edge_embeddings_0.shape[-1]
        feat1 = edge_embeddings_1.shape[-1]

        # IMPORTANT: expand instead of repeat (no real copy)
        col_0 = edges[..., 0].unsqueeze(-1).expand(-1, -1, feat0)
        col_1 = edges[..., 1].unsqueeze(-1).expand(-1, -1, feat1)

        # nodes axis is dim=1 (same as your -2)
        edge_mean_0 = scatter_mean(edge_embeddings_0, col_0, dim=1, dim_size=N)
        edge_mean_1 = scatter_mean(edge_embeddings_1, col_1, dim=1, dim_size=N)

        edge_mean = torch.cat([edge_mean_0, edge_mean_1], dim=-1)
        node_inpt = torch.cat([V, edge_mean], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings

class Decoder(nn.Module):
    def __init__(self, 
                 N = 4,   
                 enc_dim=128, 
                 enc_s_dim = 10,
                 state_size=1):
        super().__init__()
        

        self.delta_net = nn.Sequential(
            nn.Linear(N * enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, state_size)
        )

    def forward(self, V_all, pos_enc):
        
        # V_all.dim = [bs, n_block, N, enc_dim]
        # pos_enc.dim = [bs, N, enc_s_dim]
        b, n_block, N, enc_dim = V_all.shape
        V_all = V_all.permute(0, 2, 1, 3).reshape(b, N, -1)
        V = self.delta_net(torch.cat([V_all, pos_enc], dim=-1))
        
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
                 enc_c_dim = 12
                 ):
        super(Encoder, self).__init__()

        # node embedding
        self.fv1 = MLP(input_size=state_size + space_size, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_time = MLP(input_size=enc_t_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=enc_c_dim, output_size=enc_dim, act='SiLU', layer_norm=False)

        # edge embedding
        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=False)
        
    def forward(self, node_pos, state_in, time_i, conditions, edges):
        # state_in: (bs,N,in_dim), node_pos: (bs,N,space)
        
        # node embedding
        state_in = torch.cat((state_in, node_pos), dim=-1)
        time_enc = self.fv_time(time_i)         # (bs, enc_dim)
        cond_enc = self.fv_cond(conditions)     # (bs, enc_dim)
    
        V = self.fv1(state_in) + time_enc.unsqueeze(-2) + cond_enc.unsqueeze(-2)
        
        # edge embedding
        E = self.fe(get_edge_info(edges, node_pos))
        
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
                 N_block=4,
                 in_dim=4,
                 out_dim=4,
                 enc_dim=128,
                 n_head=4,
                 n_token=128,
                 dt:float =0.05,
                 stepper_scheme="euler"
                 ):
        super(Model, self).__init__()

        self.dt = dt
        self.stepper_scheme = stepper_scheme

        self.pos_enc_dim = pos_enc_dim
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 1 + 2 * pos_enc_dim
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
            enc_s_dim=enc_s_dim
            )
        
        self.decoder = Decoder(
            N=N_block, 
            enc_dim=enc_dim, 
            enc_s_dim=enc_s_dim,
            state_size=out_dim
            )

    def forward(self, state_in, node_pos, edges, time_i, conditions, pos_enc = None, c_enc = None, dt=None):
        
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
            c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)
        
        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim) # 时间编码

        edges_long = edges.long() if edges.dtype != torch.long else edges
        V, E = self.encoder(node_pos, state_in, t_enc, c_enc, edges_long)
        
        V_all = self.mixer(V, E, edges_long, pos_enc)
        
        v_pred = self.decoder(V_all, pos_enc)

        # if self.stepper_scheme == "euler":
        if dt is None: dt = self.dt  # 没有输入用默认 dt
        elif len(dt.shape) == 1: dt = dt.view(-1, 1, 1)
        state_pred = state_in + dt * v_pred

        return state_pred

    def autoregressive(self,
                       state_in,
                       node_pos,
                       edges,
                       time_seq,
                       conditions,
                       dt=None,
                       check_point=False):


        state_t = state_in
        outputs = [state_in]

        T = time_seq.shape[1]

        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)
        
        for t in range(T):
            time_i = time_seq[:, t]  # expect shape (bs, 1) or (bs,) depending on your caller

            def custom_forward(s_t, t_i):
                return self.forward(s_t, node_pos, edges, t_i, conditions, pos_enc, c_enc, dt)
            
            if check_point:
                if state_t.requires_grad == False and state_t.is_floating_point():
                    state_t.requires_grad_()

                state_t = checkpoint(custom_forward, state_t, time_i, use_reentrant=False)
            
            else:
                state_t = self.forward(state_t, node_pos, edges, time_i, conditions, pos_enc, c_enc, dt)

            outputs.append(state_t)

            # with torch.no_grad():
            #     delta = outputs[-1][:, 0] - outputs[-2][:, 0]
            #     mean_delta = delta.abs().mean().item() # 平均变化量
            #     max_delta = delta.abs().max().item()   # 最大变化量 (关注局部剧烈变化)
            #     l2_norm = torch.norm(delta).item()     # 整体变化的能量
            #     print(f"Step {t:03d} Delta -> Mean: {mean_delta:.2e} | Max: {max_delta:.2e} | L2: {l2_norm:.2e}")
        
        outputs = torch.stack(outputs[1:], dim=1)
        
        return outputs

# 均匀化
# 二维到三维演化
# 简单的算例
