import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

times = 1000  # 定义梯度下降次数
lrate = 0.01  # 记录每次梯度下降参数变化率
epoches = []  # 记录迭代过程中的索引

w0, w1, losses = [1], [1], []
for i in range(1, times + 1):
    # 求取loss函数值
    epoches.append(i)
    loss = ((w0[-1] + w1[-1] * train_x - train_y) ** 2).sum() / 2
    losses.append(loss)
    # 输出epoches wo w1 loss 状态
    print('{:4}> w0={:.8f}, w1={:.8f}, loss={:.8f}'.format(
        epoches[-1], w0[-1], w1[-1], losses[-1]))

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
mp.scatter(
    train_x,
    train_y,
    marker='s',
    c='dodgerblue',
    alpha=0.5,
    s=80,
    label='Training')
mp.plot(
    train_x,
    pred_train_y,
    '--',
    c='limegreen',
    label='Regression',
    linewidth=1)
mp.legend()

# 绘制每次梯度下降w0 w1 loss的变化曲线
w0 = w0[:-1]
w1 = w1[:-1]
mp.figure('Training Progress', facecolor='lightgray')
mp.subplot(311)
mp.title('T')
mp.ylabel('w0')
mp.plot(epoches, w0, color='dodgerblue', label='w0')
mp.grid(linestyle=':')
mp.legend()

mp.subplot(312)
mp.ylabel('w1')
mp.plot(epoches, w1, color='orangered', label='w1')
mp.grid(linestyle=':')
mp.legend()

mp.subplot(313)
mp.ylabel('loss')
mp.plot(epoches, losses, color='red', label='loss')
mp.grid(linestyle=':')
mp.legend()

# 基于三维曲面绘制梯度下降过程中每一个点
import mpl_toolkits.mplot3d as axes3d

grid_w0, grid_w1 = np.meshgrid(
    np.linspace(0,9,500),
    np.linspace(0,3.5,500),
)

grid_loss = np.zeros_like(grid_w0)
for x,y in zip(train_x,train_y):
    grid_loss += (grid_w0 + grid_w1 * x - y) ** 2 / 2

mp.figure('Loss Function')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zlabel('loss')
ax3d.plot_surface(grid_w0, grid_w1, grid_loss, rstride=30, cstride=30, cmap='jet')
ax3d.plot(w0, w1, losses, 'o-', color='orangered', zorder=3)
mp.tight_layout()

mp.show()
