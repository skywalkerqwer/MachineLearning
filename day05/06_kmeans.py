"""
k均值算法
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x= np.loadtxt('../ml_data/multiple3.txt' ,delimiter =',')

# 构建聚类模型
model = sc.KMeans(n_clusters=4)
model.fit(x)
centers = model.cluster_centers_

# 返回每个样本的聚类标签类别 0/1/2/3
pred_y = model.labels_

mp.figure('K-Means Cluster', facecolor='lightgray')
mp.title('K-Means Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
# mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=pred_y, cmap='brg', s=80)
mp.scatter(centers[:, 0], centers[:, 1], marker='+', c='red', s=300, linewidth=4)
mp.show()