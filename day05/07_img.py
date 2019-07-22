import numpy as np
import scipy.misc as sm
import sklearn.cluster as sc
import matplotlib.pyplot as mp


# 通过K均值聚类量化图像中的颜色
def quant(image, n_clusters):
    x = image.reshape(-1, 1)
    model = sc.KMeans(n_clusters=n_clusters)
    model.fit(x)
    # 获取每个像素的亮度值所属的聚类类别标签:y = 250000*(0/1/2/3)
    # y = [2,0,1,3,2,0,2,3,1,3,1,0,2,1,...]
    y = model.labels_
    # 获取聚类中心
    centers = model.cluster_centers_.ravel()
    return centers[y].reshape(image.shape)


original = sm.imread('../ml_data/lily.jpg', True)
quant4 = quant(original, 4)
quant3 = quant(original, 3)
quant2 = quant(original, 2)
mp.figure('Image Quant', facecolor='lightgray')
mp.subplot(221)
mp.title('Original', fontsize=16)
mp.axis('off')
mp.imshow(original, cmap='gray')
mp.subplot(222)
mp.title('Quant-4', fontsize=16)
mp.axis('off')
mp.imshow(quant4, cmap='gray')
mp.subplot(223)
mp.title('Quant-3', fontsize=16)
mp.axis('off')
mp.imshow(quant3, cmap='gray')
mp.subplot(224)
mp.title('Quant-2', fontsize=16)
mp.axis('off')
mp.imshow(quant2, cmap='gray')
mp.tight_layout()
mp.show()