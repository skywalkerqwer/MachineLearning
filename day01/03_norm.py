"""
归一化(正则化)
变换后的样本矩阵，每个样本的特征值绝对值之和为1。
"""
import numpy as np
import sklearn.preprocessing as sp

samples = np.array([
    [17, 100, 4000],  # 年龄  分数  工资
    [20, 80, 5000],
    [23, 60, 5500],
])
r_samples = sp.normalize(samples,norm='l2')
print(r_samples)
