import cv2 as cv

original = cv.imread( '../ml_data/chair.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow('Original', original)
# hsobel = cv.Sobel(original, cv.CV_64F, 1, 0, ksize=5)  # .CV_64代表运算精度
# cv.imshow('H-Sobel', hsobel)
# vsobel = cv.Sobel(original, cv.CV_64F, 0, 1, ksize=5)
# cv.imshow('V-Sobel', vsobel)
# sobel = cv.Sobel(original, cv.CV_64F, 1, 1, ksize=5)
# cv.imshow('Sobel', sobel)
# laplacian = cv.Laplacian(original, cv.CV_64F)
# cv.imshow('Laplacian', laplacian)
canny = cv.Canny(original, 40, 205)  # 水平和垂直方向阈值  超过阈值判断为边缘 控制精度
cv.imshow('Canny', canny)
cv.waitKey()
