import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

times = 1000	# 定义梯度下降次数
lrate = 0.01	# 记录每次梯度下降参数变化率
w0, w1 = [1], [1]
for i in range(1, times + 1):
	# d0是损失函数在w0方向上的偏导数
    d0 = (w0[-1] + w1[-1] * train_x - train_y).sum()
    # d1是损失函数在w1方向上的偏导数
    d1 = (((w0[-1] + w1[-1] * train_x) - train_y) * train_x).sum()
    # 让w0   w1不断更新  
    w0.append(w0[-1] - lrate * d0)
    w1.append(w1[-1] - lrate * d1)

pred_train_y = w0[-1] + w1[-1] * train_x
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, marker='s', c='dodgerblue', alpha=0.5, s=80, label='Training')
mp.plot(train_x, pred_train_y, '--', c='limegreen', label='Regression', linewidth=1)
mp.legend()
mp.show()