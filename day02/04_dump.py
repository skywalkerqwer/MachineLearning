"""
模型的保存
"""

import numpy as np
import sklearn.linear_model as lm
import pickle

# 采集数据
x, y = np.loadtxt('../ml_data/single.txt', delimiter=',', usecols=(0,1), unpack=True)

# 训练模型
x = x.reshape(x.size, 1)  # 把 x 变为 n 行 1 列
model = lm.LinearRegression()
model.fit(x, y)

# 保存模型
with open('../ml_data/lm.pkl','wb') as f:
    pickle.dump(model, f)

print('dump success!')