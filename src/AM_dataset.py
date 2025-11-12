import random, copy
from torch.utils.data import Dataset
from typing import List
import torch
import numpy as np 
import h5py
from pathlib import Path

class AM_Dataset(Dataset):
    """
    针对一个数据集，从中截取多个时间步
    """
    def __init__(self, 
                 h5_path: str,
                 fields: List[str],
                 horizon: int=8,
                 interval: int=10, normalize=False,):
        # super(AM_Dataset, self).__init__()

        self.h5_path = h5_path
        self.fields = fields # 一个或多个场
        self.horizon = horizon # 时间步
        self.interval = interval
        self.dt =  1e-5 # 真实时间差

        self._f = None # 存放数据
        self._points, self._cells = None, None
        self._pttype, self._cond = None, None
        # self._points_type = None
        self.do_normalization = normalize

        with h5py.File(h5_path, "r") as f:
            self.T = f['time'].shape[0]
            self.N = f['point'].shape[0] # 边数量
            self.E = f['mesh'].shape[0] # 网格数量
            self.D = f['point'].shape[1] # 维度

            max_start = self.T - 1 - self.interval * (self.horizon - 1)
            if max_start < 1:
                raise ValueError(f"文件{h5_path}时间长度不足，无法构造给定结构的样本")
            self.start_indices = np.arange(1, max_start + 1, dtype=np.int32) # 自定义窗口起点集合

    def _require(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
            self._points, self._cells, self._times = self._f['point'][:], self._f['mesh'][:], self._f['time'][:]
            self._pttype = self._f['point_type'][:]
            self._cond = {"thermal": self._f['cond']['thermal'][:], "transport": self._f['cond']['transport'], "dump": self._f['cond']['dump']}
            # 构建 edges 
            self._edges = self.make_edges(single=True)

    def __len__(self):
        return len(self.start_indices)
    
    def __getitem__(self, index):
        self._require()
        t_start = int(self.start_indices[index]) # 开始时间

        # 对应的时间步
        time_idxs = [t_start] + [t_start + i*self.interval for i in range(self.horizon)]
        times = np.array([self._times[i] for i in time_idxs]) # 输入具体的时间

        # points, cells, points_type, conds, T, time
        input = {"point": torch.from_numpy(self._points),
                 "point_type": torch.from_numpy(self._pttype),
                 "edge": copy.deepcopy(self._edges),
                 "time": torch.from_numpy(times),
                 "thermal": torch.from_numpy(self._cond['thermal']),
                 "transport": torch.from_numpy(self._cond['transport']),
                 "dump": torch.from_numpy(self._cond['dump'])}
        states = []
        for key in self.fields:
            X0 = self._f['state'][key][t_start:self.T:self.interval][:self.horizon] # (t, m, 1)
            X0 = np.asarray(X0)
            if X0.ndim == 2: # 避免遇到 v
                X0.reshape(self.horizon, self.N, None)
            states.append(X0)

        input['state'] = torch.from_numpy(np.concatenate(states, axis=-1))

        return input

    def make_edges(self, single=True):
        # 对于正六面体
        pattern = np.array([
            [0,1],[1,2],[2,3],[0,3],
            [4,5],[5,6],[6,7],[4,7],
            [0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
        cols = pattern.ravel()
        flat_edges = self._cells[:, cols]
        cell_edges = flat_edges.reshape(self._cells.shape[0], len(pattern), 2)

        edges_sorted = np.sort(cell_edges, axis=2)
        edges = edges_sorted.reshape(-1, 2) # 拉平
        edges = torch.unique(torch.from_numpy(edges).int(), dim=0) # 单向去重边

        if not single:
            edges = torch.cat([edges, torch.flip(edges, dims=[-1])], dim=0)
        return edges
    
    def normalize(self):
        """
        归一化
        """
        pass

# test
if __name__ == "__main__":
    dataset = AM_Dataset(h5_path=str(Path(r"~/MyAI/AMGTO/Dataset/Tiny_mesh_series.h5").expanduser()), fields=['T'])
    for data in dataset:
        for key in list(data.keys()):
            print(key, end=':')
            print(data[key].shape, data[key].dtype)
        break