"""
标签编码
"""
import numpy as np
import sklearn.preprocessing as sp

samples = np.array(['audi', 'ford', 'audi', 'toyota',
                    'bmw', 'toyota', 'audi', 'redflag'])

# 获取标签编码器对象
lbe = sp.LabelEncoder()
r_samples = lbe.fit_transform(samples)
print(r_samples)
# 逆向编码
result = lbe.inverse_transform(r_samples)
print(result)
