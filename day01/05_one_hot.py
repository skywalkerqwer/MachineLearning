import numpy as np
import sklearn.preprocessing as sp

samples = np.array([
    [1, 3, 2],
    [7, 5, 4],
    [1, 8, 6],
    [7, 3, 9],
])

# 构建独热编码器对象
ohe = sp.OneHotEncoder(sparse=False, dtype='i4')  # sparse=True 返回稀疏矩阵
r_samples = ohe.fit_transform(samples)
print(r_samples)
"""
r_samples: <csr_matrix>稀疏矩阵
  (0, 5)	1   -->  0,5位置元素为1
...
...
"""
# 获取码表
encoder_dict = ohe.fit(samples)
# 对不同数据使用同一个码表进行独热编码计算
r_samples = encoder_dict.transform(samples)
print(r_samples,type(r_samples))