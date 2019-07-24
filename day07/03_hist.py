import cv2 as cv

original = cv.imread('../ml_data/sunrise.jpg')
cv.imshow('Original', original)
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)  # 得到原始图像的灰度图
cv.imshow('Gray', gray)
equalized_gray = cv.equalizeHist(gray)  # 直方图均衡化
cv.imshow('Equalized Gray', equalized_gray)

# YUV：亮度，色度，饱和度
yuv = cv.cvtColor(original, cv.COLOR_BGR2YUV)
yuv[..., 0] = cv.equalizeHist(yuv[..., 0])  # [..., 0]只获取最后一个元素
equalized_color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('Equalized Color', equalized_color)
cv.waitKey()