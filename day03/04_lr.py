"""
逻辑分类
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

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

# 组合(grid_x,grid_y)为250000个坐标点作为测试集
mesh_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))

# 创建模型针对test_x预测相应输出
model = lm.LogisticRegression(solver='liblinear', C=1)
model.fit(x, y)
mesh_y = model.predict(mesh_x)

# 把预测结果变维:500*500 用于绘制边界线
grid_z = mesh_y.reshape(grid_x.shape)

# 绘制散点图
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
mp.show()
