import pyvista as pv
from pyvista.core.datasetattributes import DataSetAttributes
path = r"~/MyAI/AMGTO/Dataset/lowpower/VTK/"

file:DataSetAttributes = pv.read(path+r"lowpower_108.vtk")

print(file)
"""
UnstructuredGrid (0x71ad911d28c0)
  N Cells:    331776
  N Points:   348725
  X Bounds:   0.000e+00, 1.440e-03
  Y Bounds:   -2.900e-04, 3.500e-04
  Z Bounds:   3.200e-04, 6.800e-04
  N Arrays:   17
"""

# print(file.points[:5]) # 348725 * 3

# print(file.cells.shape) # 2985984(331776 * 9)
print(file.celltypes) # 12 六面体hex
# print(file.cells[:9]) # [   8    1 9426 9571  146    0 9425 9570  145]

print(list(file.point_data.keys())) # ['gamma_liquid', 'F_sum1', 'alpha.air', 'p_rgh', 'alpha.titanium', 'alpha.niobium', 'T', 'U']
print(type(file.point_data)) # <class 'pyvista.core.datasetattributes.DataSetAttributes'> / DataSetAttributes

print(type(file.point_data['T'][:][0]))

# print(file.point_data['U'][:].shape) # (348725,) / 速度：(348725, 3)

# print(list(file.cell_data.keys())) # ['cellID', 'gamma_liquid', 'F_sum1', 'alpha.air', 'p_rgh', 'alpha.titanium', 'alpha.niobium', 'T', 'U']

# print(file.cell_data['cellID'][:])  # (331776,)

wall_path = r"~/MyAI/AMGTO/Dataset/lowpower/VTK/rightWall/"
wall_file = pv.read(wall_path+r"rightWall_108.vtk")
print(wall_file)
"""
  N Cells:    2304  36 * 64
  N Points:   2405  37 * 65
  N Strips:   0
  X Bounds:   1.440e-03, 1.440e-03
  Y Bounds:   -2.900e-04, 3.500e-04
  Z Bounds:   3.200e-04, 6.800e-04
  N Arrays:   25
"""
print(wall_file.point_data.keys()) # 同上