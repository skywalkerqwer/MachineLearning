"""
单颗决策树
"""

import sklearn.tree as st
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.metrics as sm

# 加载数据集
boston = sd.load_boston()
# print(boston.data.shape)  # 输入集
# print(boston.target.shape)  # 输出集
# print(boston.feature_names)

# 打乱数据集
x, y = su.shuffle(boston.data, boston.target, random_state=7)

# 划分训练集与测试集
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = x[:train_size], x[train_size:],\
                                   y[:train_size], y[train_size:]

# 构建决策树模型, 训练模型
model = st.DecisionTreeRegressor(max_depth=5)  # 决策树的最大深度为 max_depth
model.fit(train_x, train_y)

# 预测
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
