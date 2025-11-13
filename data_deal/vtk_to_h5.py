"""
Transform and zip the initial vtk files to a h5 file over one property.
"""
import re, glob, numpy as np, h5py, pyvista as pv, time, os
from pyvista.core.datasetattributes import DataSetAttributes
from typing import List
from pathlib import Path
from tqdm import tqdm
from get_property import *

# 提取时间步
def extract_time(path: str) -> float:
    return float(re.findall(r"\d+", path)[-1])

def from_vtk_to_h5(name: str = "Tiny_mesh", 
                   dataset_path: str = None, 
                   interval: int = 10,
                   field: List[str] = ['T', 'F_sum1'], 
                   float32: bool = True, int32: bool = True, 
                   step: float = 1e-5):
    folder_path = os.path.abspath(os.path.dirname(__file__)) # 当前文件夹的位置
    # =====config=======
    NAME = name
    if dataset_path is not None:
        PATH = os.path.join(dataset_path, NAME)
    else: 
        PATH = os.path.join(os.path.dirname(folder_path), f"Dataset/{NAME}")
    INPUT_PATTERN = os.path.join(PATH, f"VTK/{NAME}_*.vtk")
    INTERVAL = interval
    OUT_H5 = os.path.join(os.path.dirname(folder_path), "H5set", f"{NAME}_series.h5")
    FIELDS = field
    FLOAT32, INT32 = float32, int32
    STEP = step

    # 读取文件列表
    files = sorted(glob.glob(INPUT_PATTERN), key=lambda s: extract_time(s))
    assert files, f"未找到匹配文件：{INPUT_PATTERN}" # 为空的时候触发错误
    float_type, int_type = np.float32 if FLOAT32 else np.float64, np.int32 if INT32 else np.int64
    # 基准网格（第 0 个时间步）
    ref:DataSetAttributes = pv.read(files[0])
    N, M = ref.n_points, ref.n_cells
    ref_points = ref.points.copy() # 348725 * 3
    ref_cells = ref.cells.copy() # 2985984
    celltypes = ref.celltypes.copy()

    if not (celltypes == 12).all():
        raise ValueError(f"存在非六面体的单元类型: {np.unique(celltypes)}")

    # 处理 cells
    cells_2d = ref_cells.reshape(M, 9)
    if not np.all(cells_2d[:,0] == 8):
        raise ValueError("cells 序列不是标准六面体格式。")
    cells_2d = cells_2d[:, 1:].astype(int_type)     # (331776, 8)

    pts = ref_points.astype(float_type)
    pts_type = np.zeros((ref_points.shape[0], 1), dtype=int_type)

    ctypes = celltypes.astype(np.uint8)

    # conditions 处理（激光、金属粉末）
    thermal_conds, material_conds = get_all_properties(os.path.join(PATH, "constant"), float_type=float_type)
    dump_conds = get_dumps(PATH)
    # 创建 h5 文件
    with h5py.File(OUT_H5, "w") as f:
        # mesh and points
        f.create_dataset("mesh", data=cells_2d, compression="gzip", compression_opts=4, shuffle=False)
        f.create_dataset("mesh_type", data=ctypes[:1], compression="gzip", compression_opts=4, shuffle=False)
        f.create_dataset("point", data=pts, compression="gzip", compression_opts=4, shuffle=False)    
        f.create_dataset("point_type", data=pts_type, compression="gzip", compression_opts=4, shuffle=False)
        # f.create_dataset("cond", data=conds, compression="gzip", compression_opts=4, shuffle=False)
        # f.create_dataset("point_type")

        times = f.create_dataset("time", shape=(0,), maxshape=(None,), dtype=float_type)
        g = f.create_group("state")
        conds = f.create_group('cond')
        states = {}
        for k in FIELDS:
            if k not in ref.point_data:
                raise KeyError(f"在vtk文件中未找到字段：{k}")
            data = ref.point_data[k][:]
            data = data.reshape(N, -1) # 向量变矩阵
            shape = (0, N, data.shape[1])
            states[k] = g.create_dataset(
                k, shape=shape, maxshape=(None, N, data.shape[1]), dtype=float_type,
                compression="gzip", compression_opts=4, shuffle=False,
                chunks=(1, N, data.shape[1]))
        
        # 逐个时间步写入
        for file in tqdm(files, desc="Processing", ncols=100, unit="file", colour="cyan"):
            m = pv.read(file)
            if not (m.n_points == N and m.n_cells == M) and np.array_equal(m.cells, ref_cells) and np.allclose(m.points, ref_points):
                raise ValueError(f"网格在{file}中改变！")
            
            times.resize((times.shape[0]+1,))
            times[-1] = (times.shape[0] - 1) * STEP

            for key, dset in states.items():
                arr = m.point_data[key][:] # 拷贝数据
                arr = arr.reshape(N, -1)
                dset.resize((dset.shape[0]+1, N, arr.shape[1]))
                dset[-1] = arr

        # 写入条件
        conds.create_dataset("thermal", data=thermal_conds, compression="gzip", compression_opts=4, shuffle=False)
        conds.create_dataset("material", data=material_conds, compression="gzip", compression_opts=4, shuffle=False)
        conds.create_dataset("dump", data=dump_conds, compression="gzip", compression_opts=4, shuffle=False)

    print(f"✅ 转换完成：{OUT_H5}")

from_vtk_to_h5()


