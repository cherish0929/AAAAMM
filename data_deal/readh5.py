import h5py, torch

file = h5py.File('/home/ubuntu/MyAI/AMGTO/Dataset/Tiny_mesh_series.h5')

# print(list(file.keys()))

# print(list(file['state'].keys()))

# print(file['mesh_type'])
# print(file['mesh'].shape)
for cell in file['mesh'][:]:
    for i in cell:
        print(i,file['point'][i])
    break
# print(file['point'].shape)
# print(file['state']['F_sum1'].shape)
# print(file['state']['time'].shape)

# print(max([m[0] for m in file['node']['node_pos'][:]]), min([m[0] for m in file['node']['node_pos'][:]]))
# print(max([m[1] for m in file['node']['node_pos'][:]]), min([m[1] for m in file['node']['node_pos'][:]]))

# for n in range(3424):
#     if file['node']['node_type'][:][n] == 4.0:
#         print(file['node']['node_pos'][:][n])

# print(set(file['node']['node_type'][:][:10]))
# print(list(file['node']['node_type'][:]).count(0.0))
# print(list(file['node']['node_type'][:]).count(1.0))
# print(list(file['node']['node_type'][:]).count(4.0))

# print(max([max(m) for m in file['mesh'][:]]))

# print(file['state']['ne'][:][0,0,:10])
# print(type(file['state']['T'][:][0][0][0]))  # float32

# print(file['state']['t'][:][::5])
# """
# [0.00000000e+00 7.94328235e-08 7.94328235e-07 7.94328235e-06
#  7.94328235e-05 7.94328235e-04 7.94328235e-03 7.94328235e-02
#  7.94328235e-01]
# """

# print(file['state']['num_of_density'].keys())

# for key in list(file.keys()):
#     data = file[key]
#     print(key, end=":")
#     if isinstance(data, h5py.Dataset):
#         # print(data.shape, data.dtype)
#         continue
#     elif isinstance(data, h5py.Group):
#         print(list(data.keys()))
#         for ke in list(data.keys()):
#             print(f"{ke}: {type(data[ke])}")
#             if isinstance(data[ke], h5py.Dataset):
#                 print(data[ke].shape, data[ke].dtype)
#     print("---------------")

"""
conditions: 7

mesh: 20422 * 3
[606 607 608]
[606 607 608]
[606 607 608]
[606 608 609]
[606 608 609]

node_pos / node_type 3424

T/ne/te/v 1 * 82 * 3424

82个时间步；3424个位置

t:
[0.00000000e+00 2.51188643e-08 7.94328235e-08 2.51188643e-07
 7.94328235e-07 2.51188643e-06 7.94328235e-06 2.51188643e-05
 7.94328235e-05 2.51188643e-04 7.94328235e-04 2.51188643e-03
 7.94328235e-03 2.51188643e-02 7.94328235e-02 2.51188643e-01
 7.94328235e-01]

pos:
[0.01000163 0.25      ]
[0.01499837 0.25      ]
[0.01000163 0.25507814]
[0.01499837 0.25507814]
[0.00500081 0.25      ]

type 0, 1, 4
"""