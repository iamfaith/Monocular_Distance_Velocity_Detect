excel_path = r'./camera_parameters.xlsx'
import pandas as pd
import numpy as np
import math
def camera_parameters(excel_path):
    # Load Intrinsics matrix of Camera
    df_intrinsic = pd.read_excel(excel_path, sheet_name='内参矩阵', header=None)
    # Load Extrinsics matrix of Camera
    df_p = pd.read_excel(excel_path, sheet_name='外参矩阵', header=None)

    print('外参矩阵形状 intrinsics matrix shape：', df_p.values.shape)
    print('内参矩阵形状 Extrinsics matrix shape：', df_intrinsic.values.shape)

    return df_p.values, df_intrinsic.values



p, k = camera_parameters(excel_path) #p外参矩阵, k内参矩阵
print(p)
print(k)

u1 = 20
v1 = 10 #v + h / 2
fy = k[1, 1]
H = 40

Height = 0.5

#The angle between the camera len and the horizontal line(the moving direction of vehicle), default is 0
#相机与水平线夹角, 默认为0 相机镜头正对前方，无倾斜
angle_a = 0
angle_b = math.atan((v1 - H / 2) / fy)
angle_c = angle_b + angle_a
print('angle_b', angle_b)

depth = (Height / np.sin(angle_c)) * math.cos(angle_b)
print('depth', depth)
    
    
p_inv = np.linalg.inv(p)
print(p_inv)
k_inv = np.linalg.inv(k)

point_c = np.array([u1, v1, 1])
point_c = np.transpose(point_c)
#point (x,y) in camera coordinate position
c_position = np.matmul(k_inv, depth * point_c)
print('相机坐标系camera_coordinate_position', c_position)

# c = p[R|t] * w --> w = p^-1[R|t] *c
#point (x,y) in world coordinate position
c_position = np.append(c_position, 1)
c_position = np.transpose(c_position) # [20 10  1  1]
c_position = np.matmul(p_inv, c_position)
print('世界坐标系world_coordinate_position', c_position)
print(c_position)
# 相机坐标系camera_coordinate_position [ 30.62056573  17.174715   -49.799395  ]
# 世界坐标系world_coordinate_position [-49.799395    30.62056573  16.174715     1.        ]

# 世界x --》 相机z
# 世界y --》 相机x
# 世界z --》 相机y - 1
# [[0 1 0 0]
#  [0 0 1 1]
#  [1 0 0 0]
#  [0 0 0 1]]

            #       Zw
            #     |
            #     |
            #     |
            #     |________ Yw
            #    /
            #   /    .(obj) 
            #  /
            # Xw         | Yc
            #            |___ Xc
            #           /
            #          / Zc