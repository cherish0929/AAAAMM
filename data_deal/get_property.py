"""
读取 constant 下的文件
提取激光、材料等参数
"""
import os
import re
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, List, Union

# ========== 基础解析工具：去注释 ==========
def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # 块注释
    text = re.sub(r"//.*", "", text)                  # 行注释
    return text

# ========== 行匹配（支持三种写法）==========
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PAT_DUP  = re.compile(rf"^(?P<key>\w+)\s+\w+\s*(?:\[[^\]]*\])\s*(?P<val>{_NUM})\s*;?$")
_PAT_DIM  = re.compile(rf"^(?P<key>\w+)\s*(?:\[[^\]]*\])\s*(?P<val>{_NUM})\s*;?$")
_PAT_SIMP = re.compile(rf"^(?P<key>\w+)\s*(?P<val>{_NUM})\s*;?$")

_SKIP_PREFIXES = ("FoamFile", "version", "format", "class", "location", "object")

def _parse_block_lines(block: str) -> "OrderedDict[str, float]":
    """解析一个扁平块中的标量项（忽略维度），返回有序字典 {key: float(value)}。"""
    out = OrderedDict()
    for raw in block.splitlines():
        ln = raw.strip()
        if not ln or ln in ("{", "}", "(", ")") or ln.startswith(_SKIP_PREFIXES):
            continue
        m = _PAT_DUP.match(ln) or _PAT_DIM.match(ln)
        if not m and ("[" not in ln and "]" not in ln):
            m = _PAT_SIMP.match(ln)
        if m:
            key = m.group("key")
            try:
                val = float(m.group("val"))
            except ValueError:
                continue
            out[key] = val
    return out

def parse_openfoam_file(path: str,
                        flatten_phases: bool = True,
                        phase_sep: str = ".") -> Dict[str, Union[float, Dict[str, float]]]:
    """
    解析 OpenFOAM 字典文件，提取“变量名 → 数值”。
    - 存在 phases(...)：
        * flatten_phases=True  -> {'titanium.nu': 8e-07, 'titanium.rho': 7400, ...}
        * flatten_phases=False -> {'titanium': {'nu': 8e-07, 'rho': 7400, ...}, ...}
    - 不存在 phases          -> 扁平 {key: value}
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = _strip_comments(f.read())

    # phases(...) 主体
    m_phases = re.search(r"phases\s*\(\s*(?P<body>.*?)\s*\)", text, flags=re.S)
    if not m_phases:
        # 非多相：直接解析整文件扁平化
        return _parse_block_lines(text)

    # 抽出材料块
    phases_body = m_phases.group("body")
    phase_blocks = re.findall(r"(\w+)\s*\{([^{}]*)\}", phases_body, flags=re.S)

    if flatten_phases:
        flat = OrderedDict()
        for phase_name, block in phase_blocks:
            props = _parse_block_lines(block)
            for k, v in props.items():
                flat[f"{phase_name}{phase_sep}{k}"] = v
        # 若文件里还有 phases 外的顶层键（如激光参数），也一起抓
        # 删除 phases(...)，再解析剩余
        rest = text.replace(m_phases.group(0), "")
        rest_props = _parse_block_lines(rest)
        flat.update(rest_props)
        return flat
    else:
        nested = OrderedDict()
        for phase_name, block in phase_blocks:
            nested[phase_name] = _parse_block_lines(block)
        return nested

def get_all_properties(constant_path: str,
                       want_material: bool = True,
                       fill_value: float = -1.0,
                       float_type = np.float32,
                      ):
    """
    - 读取 constant/thermalProperties，提取**非 phases**的全局/激光参数（按键名排序对齐）。
    - 读取 constant/transportProperties（若存在 phases），将**各材料参数**按“列对齐”，缺失填 fill_value。
    - 返回：
        thermal_vec, thermal_keys, material_mat, material_keys, material_names
      其中 material_* 可能为 None（当 want_material=False 或文件不存在/无材料）。
    """
    thermal_path   = os.path.join(constant_path, "thermalProperties")
    material_path = os.path.join(constant_path, "transportProperties")
    
    # ==== 1) thermal：只要顶层键（比如激光/全局），排除 phases.* ====
    if os.path.exists(thermal_path):
        # 先拿到“扁平且含 phases.* 的全集”
        tp_full = parse_openfoam_file(thermal_path, flatten_phases=True)

    thermal_vec = np.array(list(tp_full.values()), dtype=float_type) if tp_full else np.zeros((0,), dtype=float_type)

    # ==== 2) material：对齐各材料的列 ====
    material_mat = None
    material_keys = None
    material_names = None

    if want_material and os.path.exists(material_path):
        trans = parse_openfoam_file(material_path, flatten_phases=False)  # {material: {k: v}}
        if isinstance(trans, dict) and trans:
            # 收集所有列（参数名）并排序，保证对齐 & 可复现
            all_keys = {k for props in trans.values() for k in props.keys()}
            materials = list(trans.keys())  # 行顺序：按材料名排序
        
            mat = np.full((len(materials), len(all_keys)), fill_value, dtype=float_type)
            for i, m in enumerate(materials):
                props = trans[m]
                for j, key in enumerate(all_keys):
                    if key in props:
                        try:
                            mat[i, j] = float_type(props[key])
                        except Exception:
                            mat[i, j] = float_type(fill_value)

            material_mat = mat
            # material_keys = all_keys
            # material_names = materials

    return thermal_vec, material_mat

def get_dumps(path):
    """
    用于获取文件夹下的粉末床数据
    """
    data = np.loadtxt(os.path.join(path, "dump.txt"), dtype=np.float32)

    dump_num, dump_type = len(np.unique(data)), None
    if dump_num == 1:
        dump_type = "single"
    else:
        dump_type = "multi"
    return data

# -------------------- 示例 --------------------
# if __name__ == "__main__":
#     const_dir = "/home/ubuntu/MyAI/AMGTO/Dataset/Tiny_mesh/constant"  # 改成你的路径
#     thermal_vec, material_mat, material_keys, material_names = get_all_properties(
#         const_dir, want_material=True, fill_value=-1.0
#     )

#     print("== Thermal / Global keys ==")
#     print(thermal_vec.shape, thermal_vec)

#     print("\n== material / Materials ==")
#     if material_mat is not None:
#         print("materials:", material_names)
#         print("columns  :", material_keys)
#         print("matrix   :", material_mat.shape)
#         print(material_mat)
#     else:
#         print("No material data found.")