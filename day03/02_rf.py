"""
随机森林
"""

import numpy as np
import matplotlib.pyplot as mp
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm

# 读取数据集
headers = None
data = []
with open('../ml_data/bike_day.csv', 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            headers = line.split(',')[2:]
        else:
            data.append(line.split(',')[2:])

headers = np.array(headers)
data = np.array(data, dtype='f8')

# 整理数据集
x = data[:, 0:12]
y = data[:, -1]  # 最后一列
x, y = su.shuffle(x, y, random_state=7)

# 拆分测试集与训练集
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = x[:train_size], x[train_size:],\
                                   y[:train_size], y[train_size:]

# 构建随机森林回归器模型 训练模型
model = se.RandomForestRegressor(
    max_depth=10,
    n_estimators=1000,
    min_samples_split=2)
model.fit(train_x, train_y)

# 针对测试集进行模型预测 输出评估得分
pred_y = model.predict(test_x)
print('r2:', sm.r2_score(test_y, pred_y))
