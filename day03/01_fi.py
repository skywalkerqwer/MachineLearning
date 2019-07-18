import sklearn.tree as st
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import numpy as np
import sklearn.ensemble as se

# 加载数据集
boston = sd.load_boston()
print(boston.feature_names)
feature_names = boston.feature_names

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

# 输出单棵决策树的特征重要性
dt_fi = model.feature_importances_
# print(dt_fi)

# 构建正向激励决策树模型
model = se.AdaBoostRegressor(
    model,
    n_estimators=400,
    random_state=7)  # 构建 n_estimators 棵不同权重的决策树
model.fit(train_x, train_y)
ad_fi = model.feature_importances_

# 预测
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 柱状图显示特征重要新
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('Decision Tree', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = dt_fi.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(
    pos,
    dt_fi[sorted_indices],
    0.8,
    facecolor='deepskyblue',
    edgecolor='steelblue',
    label='DT FI')
mp.xticks(pos, feature_names[sorted_indices], rotation=30)
mp.legend()

# 绘制正向激励
mp.subplot(212)
mp.title('AdaBoost Decision Tree', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = ad_fi.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(
    pos,
    ad_fi[sorted_indices],
    facecolor='lightcoral',
    edgecolor='indianred',
    label='AD FI'
)
mp.xticks(pos, feature_names[sorted_indices], rotation=30)

mp.legend()
mp.tight_layout()
mp.show()
