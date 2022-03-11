'''
    1、获取像素个数
    2、计算前景背景的灰度均值
    3、迭代比较

'''
import cv2
import numpy as np
import time


# 获取各个像素的个数
def getPixelNumber(gray):
    p_arr = np.zeros([256, ])
    gray = gray.ravel()
    for p in gray:
        p_arr[p] += 1
    return p_arr


# 计算前景背景的灰度均值
def getFBMeanGrayLevel(arr, t):
    FM, BM = 0, 0
    fl, bl = 1, 1
    for i, p in enumerate(arr):
        if i < t:
            FM += i*p
            fl += p
        else:
            BM += i*p
            bl += p
    FM /= fl
    BM /= bl
    return FM, BM


# 迭代获取最优阈值
def getOptimalThreshold(gray):
    p_arr = getPixelNumber(gray)
    best_threshold = 127
    while True:
        FM, BM = getFBMeanGrayLevel(p_arr, best_threshold)
        if best_threshold == (FM+BM) // 2:
            break
        best_threshold = (FM+BM) // 2
    return int(best_threshold)


# 读取图片
image = cv2.imread('lena.png')
# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 获取最优阈值
t1 = time.time()
T = getOptimalThreshold(gray)
print("using time", time.time()-t1)
ret, thresh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
# 可视化 灰度图
cv2.imshow("gray image", gray)
# 可视化 阈值分割图
cv2.imshow("threshold image", thresh)
cv2.waitKey()
