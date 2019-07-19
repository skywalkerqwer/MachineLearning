"""
决策树分类
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

# 读取文件
data = np.loadtxt('../ml_data/car.txt', delimiter=',', dtype='U20')
# print(data.shape)

# 整理训练集的输入输出
data = data.T
train_x, train_y = [], []
encoders = []
for col in range(len(data)):
    lbe = sp.LabelEncoder()
    if col < len(data) - 1:  # 不是最后一列
        train_x.append(lbe.fit_transform(data[col]))
    else:
        train_y = lbe.fit_transform(data[col])
    encoders.append(lbe)  # 保存每列的标签编码器
train_x = np.array(train_x).T
# print(train_x.shape)

# 交叉验证 训练模型
model = se.RandomForestClassifier(
    max_depth=6,
    n_estimators=200,
    random_state=7)  # 随机森林分类器
cv = ms.cross_val_score(model, train_x, train_y, cv=5, scoring='f1_weighted')
print(cv.mean())

model.fit(train_x, train_y)

# 模型测试
data = [
    ['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
    ['high', 'high', '4', '4', 'med', 'med', 'acc'],
    ['low', 'low', '2', '4', 'small', 'high', 'good'],
    ['low', 'med', '3', '4', 'med', 'high', 'vgood']]

data = np.array(data).T
test_x, test_y = [], []
for col in range(len(data)):
    encoder = encoders[col]
    if col < len(data) - 1:
        test_x.append(encoder.transform(data[col]))
    else:
        test_y = encoder.transform(data[col])
test_x = np.array(test_x).T
pred_y = model.predict(test_x)
print(pred_y)
print(test_y)
