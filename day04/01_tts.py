"""
训练集测试集划分
"""

import sklearn.naive_bayes as nb
import numpy as np
import matplotlib.pyplot as mp
import sklearn.model_selection as ms

data = np.loadtxt(
    '../ml_data/multiple1.txt',
    delimiter=',',
    usecols=(
        0,
        1,
        2),
    unpack=False)
x = data[:, :2]
y = data[:, 2]

# 划分
train_x, test_x, train_y, test_y = ms.train_test_split(
    x, y, test_size=0.25, random_state=7)

# 训练模型
model = nb.GaussianNB()

# 基于新创建的模型进行交叉验证
# 精确度
ac = ms.cross_val_score( model, train_x, train_y, cv=5, scoring='accuracy')
print('精确度:',ac.mean())
# 查准率
pw = ms.cross_val_score( model, train_x, train_y, cv=5, scoring='precision_weighted')
print('查准率:',pw.mean())
# 召回率
rw = ms.cross_val_score( model, train_x, train_y, cv=5, scoring='recall_weighted')
print('召回率:',rw.mean())
# f1得分
fw = ms.cross_val_score( model, train_x, train_y, cv=5, scoring='f1_weighted')
print('f1得分:',fw.mean())
model.fit(train_x, train_y)

# 把训练好的模型应用于测试集,输出精准度
pred_y = model.predict(test_x)
print(pred_y[pred_y == test_y].size / pred_y.size)

# 整理数据集, 绘制背景颜色
# 把整个空间分为500*500的网格矩阵
n = 1000
l, r = x[:, 0].min(), x[:, 0].max()  # 确定左右边界
b, t = x[:, 1].min(), x[:, 1].max()  # 确定上下边界
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n),
                             np.linspace(b, t, n))

# 组合(grid_x,grid_y)为250000个坐标点作为测试集
mesh_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
mesh_y = model.predict(mesh_x)
grid_z = mesh_y.reshape(grid_x.shape)

# 绘制所有样本点
mp.figure('Naive Bayes Classification', facecolor='lightgray')
mp.title('Naive Bayes Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap='brg', s=80)
mp.show()
