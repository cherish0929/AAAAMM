import torch, datetime
import torch.nn as nn
import numpy as np
import random
import json, sys, os
import argparse
from types import SimpleNamespace

class Logger:
    def __init__(self, filename: str, stream=sys.stdout, auto_flush=True, add_header=True):
        self.terminal = stream
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'a')
        self.auto_flush = auto_flush

        if add_header:
            self._write_header()

    def _write_header(self):
        """写入日志分隔头"""
        header = f"\n{'=' * 80}\n[LOG START @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n{'=' * 80}\n"
        self.log.write(header)
        self.terminal.write(header)
        if self.auto_flush:
            self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """与 sys.stdout 接口兼容"""
        if self.auto_flush:
            self.log.flush()

def set_seed(seed: int = 0):    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def collate(X):
    """ Convoluted function to stack simulations together in a batch. Basically, we add ghost nodes
    and ghost edges so that each sim has the same dim. This is useless when batchsize=1 though..."""
    
    # point torch.Size([1, 1954, 3])
    # edge torch.Size([1, 5628, 2]) 编号
    # time torch.Size([1, 6, 1954, 3]) 时间
    # node_type torch.Size([1, 1954, 1])
        
    N_max = max([x["point"].shape[-2] for x in X]) # point 数量
    E_max = max([x["edge"].shape[-2] for x in X]) # edge 数量
    M_max = max([x["material"].shape[-2] for x in X]) # 材料数量，不过暂时用不到
    D_max = max([x["material"].shape[-2] for x in X]) # 粉末数量
    # cells_max = max([x["cells"].shape[-2] for x in X])
    for batch, x in enumerate(X):
        # This step add fantom nodes to reach N_max + 1 nodes
        key = "state"
        tensor = x[key]
        T, N, S = tensor.shape
        x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)
                
        key = "point"
        tensor = x[key]
        if len(tensor.shape)==2: # 没有时间维度
            N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(N_max - N + 1, S)], dim=0)
        elif len(tensor.shape)==3:
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)
            
        # 参数维度都相同，无需额外的操作
        
        key = "point_type"
        tensor = x[key]
        if len(tensor.shape)==2:
            N, S = tensor.shape
            x[key] = torch.cat([tensor, 2 * torch.ones(N_max - N + 1, S)], dim=0)
        elif len(tensor.shape)==3:
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, 2 * torch.ones(T, N_max - N + 1, S)], dim=1)

        key = "edge"
        edges = x[key]
        if len(tensor.shape)==2:
            E, S = edges.shape
            x[key] = torch.cat([edges, N_max * torch.ones(E_max - E + 1, S)], dim=0)
        elif len(tensor.shape)==3:
            T, E, S = edges.shape
            x[key] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)

        x['mask'] = torch.cat([torch.ones(N), torch.zeros(N_max - N + 1)], dim=0)  # 新加入的点没有用，mask 掉


    output = {key: None for key in X[0].keys()}
    for key in output.keys():
        output[key] = torch.stack([x[key] for x in X], dim=0)

    return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None and m.bias.numel() > 0:
            m.bias.data.fill_(0.01)
    
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.in_proj_weight)
        
        if m.in_proj_bias is not None and m.in_proj_bias.numel() > 0:
            m.in_proj_bias.data.fill_(0.01)

        if m.out_proj.weight.numel() > 0:
            torch.nn.init.xavier_uniform_(m.out_proj.weight)
        
        if m.out_proj.bias is not None and m.out_proj.bias.numel() > 0:
            m.out_proj.bias.data.fill_(0.01)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    args = SimpleNamespace(**config)
    
    return args




