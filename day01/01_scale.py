"""
均值移除(标准化)
让样本矩阵中的每一列的平均值为0，标准差为1
"""
import numpy as np
import sklearn.preprocessing as sp

samples = np.array([
    [17, 100, 4000],  # 年龄  分数  工资
    [20, 80, 5000],
    [23, 60, 5500],
])

r_samples = sp.scale(samples)
print(r_samples)
print(r_samples.mean(axis=0))
print(r_samples.std(axis=0))
