"""
DBSCAN算法
"""
import numpy as np
import sklearn.cluster as sc
import sklearn.metrics as sm
import matplotlib.pyplot as mp

x = np.loadtxt('../ml_data/perf.txt', delimiter=',')
epsilons = np.linspace(0.3, 1.2, 10)
scores, models = [], []

for epsilon in epsilons:
    # DBSCAN聚类器
    model = sc.DBSCAN(eps=epsilon, min_samples=5)  # 半径、最小样本数
    model.fit(x)
    score = sm.silhouette_score(x,
                                model.labels_,
                                sample_size=len(x),
                                metric='euclidean')
    scores.append(score)
    models.append(model)

scores = np.array(scores)
best_index = scores.argmax()  # 得到最高分模型的索引
best_epsilon = epsilons[best_index]
print(best_epsilon)
best_score = scores[best_index]
print(best_score)
best_model = models[best_index]

# 获取核心样本、外周样本、孤立样本。并且使用不同的点型绘图。
pred_y = best_model.fit_predict(x)
core_mask = np.zeros(len(x), dtype=bool)
core_mask[best_model.core_sample_indices_] = True

# 对于dbscan,会出现-1的情况,label=-1时代表为孤立样本
offset_mask = best_model.labels_ == -1

# ”~“代表按位取反，目的是得到既不是核心也不是孤立的样本 -> 得到外周样本
periphery_mask = ~(core_mask | offset_mask)

mp.figure('DBSCAN Cluster', facecolor='lightgray')
mp.title('DBSCAN Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
labels = best_model.labels_
# 绘制核心样本
mp.scatter(x[core_mask][:, 0], 
           x[core_mask][:, 1],
           c=labels[core_mask],
           cmap='brg', s=80, label='Core')
# 绘制外周样本
mp.scatter(x[periphery_mask][:, 0],
           x[periphery_mask][:, 1], alpha=0.5,
           c=labels[periphery_mask],
           cmap='brg', marker='s', s=80, label='Periphery')
# 绘制孤立样本
mp.scatter(x[offset_mask][:, 0],
           x[offset_mask][:, 1],
           c=labels[offset_mask],
           cmap='brg', marker='x', s=80, label='Offset')
mp.legend()
mp.show()
