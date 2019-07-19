"""
07_svm.py  支持向量机
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/multiple2.txt',
                  delimiter=',', dtype='f8')
x = data[:, :-1]
y = data[:, -1]
# 选择svm做分类
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
                        random_state=5)
model = svm.SVC(kernel='rbf')
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))

# 绘制分类边界线
n = 500
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n),
                     np.linspace(b, t, n))
flat_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
flat_y = model.predict(flat_x)
grid_z = flat_y.reshape(grid_x.shape)

mp.figure('SVM Linear Classification', facecolor='lightgray')
mp.title('SVM Linear Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z,
              cmap='gray')
mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap='brg', s=80)
mp.show()
