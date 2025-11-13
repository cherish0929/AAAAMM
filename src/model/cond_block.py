import torch, torch.nn as nn
import torch.nn.functional as F
from .block import MLP
from typing import Optional, Tuple

class CondAttenPool(nn.Module):
    """
    material and dump
    """
    def __init__(self, d_model, d_hidden: Optional[int] = None):
        super(CondAttenPool, self).__init__()
        if d_hidden is None:
            d_hidden = d_model
        self.proj = nn.Linear(d_model, d_hidden)
        self.v = nn.Linear(d_hidden, 1, bias=False)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # H: [B, N, d] N 不确定, d 确定
        scores = self.v(torch.tanh(self.proj(H))).squeeze(-1) # [B, N]
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask > 0
            scores = scores.masked_fill(~mask, -1e9)
            valid = (mask.sum(dim=1, keepdim=True) > 0).float()  # [B,1]
        else:
            valid = torch.ones(H.size(0), 1, device=H.device, dtype=H.dtype)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)          # [B,N,1]
        pooled = (w * H).sum(dim=1)                             # [B,d]
        return pooled * valid                                   # 全 padding 时置零，避免 NaN
    
class CondEncoder(nn.Module):
    """
    逐个元素 MLP -> 注意力池化 -> MLP 投影
    """
    def __init__(self, in_dim: int,
                 N_hid: int = 2, hid_size: int =128, out_dim: int = 128):
        super(CondEncoder, self).__init__()
        # 编码
        self.phi = MLP(input_size=in_dim, output_size=hid_size,
                       n_hidden=N_hid, hidden_size=hid_size, layer_norm=False)
        self.pool = CondAttenPool(hid_size)

        # 投影
        self.rho = MLP(input_size=hid_size, output_size=out_dim, n_hidden=N_hid,
                       hidden_size=hid_size, layer_norm=False)
        
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # X: [B,N,in_dim]
        H = self.phi(X)              # [B,N,elem_hidden]
        z = self.pool(H, mask)       # [B,elem_hidden]
        z = self.rho(z)              # [B,out_dim]
        return z
    
class CondBlock(nn.Module):
    def __init__(self, 
                 dump_dim: int=5,
                 material_dim: int=15,
                 thermal_dim: int=16,
                 enc_dim: int=128,
                 head_hidden: int=256,
                 out_dim: int=10):
        super(CondBlock, self).__init__()

        self.dump_enc = CondEncoder(in_dim=dump_dim, hid_size=enc_dim,
                                      out_dim=enc_dim)
        self.material_enc = CondEncoder(in_dim=material_dim, hid_size=enc_dim,
                                        out_dim=enc_dim)
        self.thermal_enc = MLP(input_size=thermal_dim, output_size=enc_dim,
                               n_hidden=2, hidden_size=enc_dim, layer_norm=False)
        
        # 融合编码
        enc_in_dim = enc_dim * 3
        self.head = MLP(input_size=enc_in_dim, output_size=out_dim,
                        n_hidden=4, hidden_size=head_hidden, layer_norm=False)
        
        # Optional
        self.norm_dump   = nn.LayerNorm(dump_dim)
        self.norm_material = nn.LayerNorm(material_dim)
        self.norm_thermal  = nn.LayerNorm(thermal_dim)

    def forward(self,
                dump: torch.Tensor,                # [B,m,5]
                material: torch.Tensor,              # [B,n,15]
                thermal: torch.Tensor,                 # [B,20]
                dump_mask: Optional[torch.Tensor] = None,  # [B,m]  True=有效
                material_mask: Optional[torch.Tensor] = None # [B,n]
                ) -> Tuple[torch.Tensor, dict]:
        # 归一化（逐通道），避免不同量纲差异过大
        dump   = self.norm_dump(dump)
        material = self.norm_material(material)
        thermal  = self.norm_thermal(thermal)

        # 编码
        zp = self.dump_enc(dump, dump_mask)      # [B,enc_dim]
        zm = self.material_enc(material, material_mask)# [B,enc_dim]
        ze = self.thermal_enc(thermal)                     # [B,extra_enc_dim]

        x = torch.cat([zp, zm, ze], dim=-1)            # [B, enc_dim*2 + extra_enc_dim]
        y = self.head(x)                                # [B,out_dim]
        # aux = {"zp": zp, "zm": zm, "ze": ze, "fusion": x}
        return y.unsqueeze(-1)
    


            
