"""
范围缩放
把特征值缩放至[0, 1]区间
只保留数据间比例
"""

import numpy as np
import sklearn.preprocessing as sp

samples = np.array([
    [17, 100, 4000],  # 年龄  分数  工资
    [20, 80, 5000],
    [23, 60, 5500],
])

mms = sp.MinMaxScaler(feature_range=(0, 1))  # 创建MinMax缩放器
r_samples = mms.fit_transform(samples)  # 调用mms对象的方法
print(r_samples)

"""
手动实现范围缩放
"""
samples = samples.astype('float64')
for col in samples.T:
    col_min = col.min()
    col_max = col.max()
    A = np.array([
        [col_min, 1],
        [col_max, 1],
    ])
    B = np.array([0, 1])
    X = np.linalg.lstsq(A, B)[0]
    col *= X[0]  # kx
    col += X[1]  # +b

print(samples)
