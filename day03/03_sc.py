"""
人工分类
"""

import numpy as np
import matplotlib.pyplot as mp

x = np.array([
    [3, 1],
    [2, 5],
    [1, 8],
    [6, 4],
    [5, 2],
    [3, 5],
    [4, 7],
    [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 把整个空间分为500*500的网格矩阵
l, r = x[:, 0].min(), x[:, 0].max()  # 确定左右边界
b, t = x[:, 1].min(), x[:, 1].max()  # 确定上下边界
grid_x, grid_y = np.meshgrid(np.linspace(l, r, 300),
                             np.linspace(b, t, 500))

# 网格化矩阵中每个坐标点的类别标签
grid_z = np.piecewise(grid_x,
                      [grid_x > grid_y, grid_x < grid_y],
                      [0, 1])  # x>y属于0类别, x<y属于1类别


# 绘制散点图
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
#
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
mp.show()
