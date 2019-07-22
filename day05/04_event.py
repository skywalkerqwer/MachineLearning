"""
事件预测
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.metrics as sm
"""
1. 读文件
整理二维数组: data  shape(5040,5)
"""
data = []
with open('../ml_data/event.txt', 'r') as f:
    for line in f.readlines():
        data.append(line.split(','))
data = np.array(data)
data = np.delete(data, 1, axis=1)
# print(data.shape)


"""
2. 解析data, 数据预处理, 整理输入集与输出集
                  x:(5040,1)  y:(5040,1)
非数字字符串特征需要做标签编码
数字字符串特征需要做转换编码  str -> int
自定义标签编码类
"""


class DigitEncoder():
    """模拟LabelEncoder编写的数字编码器"""

    def fit_transform(self, y):
        return y.astype('i4')  # 转换为int4类型

    def transform(self, y):
        return y.astype('i4')

    def inverse_transform(self, y):
        return y.astype('str')


x, y = [], []
encoders = []
cols = data.shape[1]  # 获取一共有多少列
for i in range(cols):
    col = data[:, i]
    # 判断当前列是否是数字字符串
    if col[0].isdigit():
        encoder = DigitEncoder()
    else:
        encoder = sp.LabelEncoder()
    # 使用标签编码器
    encoders.append(encoder)
    if i < cols - 1:
        x.append(encoder.fit_transform(col))
    else:
        y = encoder.fit_transform(col)
    encoders.append(encoder)
x = np.array(x).T
y = np.array(y)
# print(x, y)

"""
3. 拆分测试集与训练集
"""
train_x, test_x, train_y, test_y = ms.train_test_split(x, y,
                                                       test_size=0.25,
                                                       random_state=7)
"""
4. 构建svm模型训练模型  使用测试集进行测试  调参
"""
model = svm.SVC(kernel='rbf',C=1,gamma=1,class_weight='balanced')

# 根据网格搜索选择最优模型
# params = [{'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001]}]  # C:1 g:1


# # 调用网格搜索获得最优模型
# model = ms.GridSearchCV(model, params, cv=5)  # 进行5次交叉验证
model.fit(train_x, train_y)
#
# for param, score in zip(model.cv_results_['params'],
#                         model.cv_results_['mean_test_score']):
#     print(param, '>>', score)

"""
5. 模型评估
"""
pred_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_y))

"""
6. 业务应用
"""
data = [['Tuesday', '13:30:00', '21', '23']]
data = np.array(data).T
x = []
for row in range(len(data)):
    encoder = encoders[row]
    x.append(encoder.transform(data[row]))
x = np.array(x).T
pred_y = model.predict(x)
print(encoders[-1].inverse_transform(pred_y))
