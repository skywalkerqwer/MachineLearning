"""
混淆矩阵
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
model.fit(x, y)
# 把训练好的模型应用于测试集
pred_y = model.predict(test_x)

# 输出混淆矩阵
import sklearn.metrics as sm
cm = sm.confusion_matrix(test_y, pred_y)
print(cm)
