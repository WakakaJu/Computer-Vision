'''
    1、获取像素分布概率
    2、前景背景分离
    3、计算当前信息熵
    4、遍历
'''
import cv2
import numpy as np
import time


# 获取各个像素的概率
def getPixelProbability(gray):
    p_arr = np.zeros([256, ])
    gray = gray.ravel()
    for p in gray:
        p_arr[p] += 1
    return p_arr / len(gray)


# 分离前景和背景
def separateForegroundAndBackground(arr, t):
    f_arr, b_arr = arr[:t], arr[t:]
    return f_arr, b_arr


# 获取某阈值下的信息熵
def informationEntropy(arr):
    Pn = np.sum(arr)
    if Pn == 0:
        return 0
    iE = 0
    for p_i in arr:
        if p_i == 0:
            continue
        iE += (p_i / Pn) * np.log(p_i / Pn)
    return -iE


# 获取最优阈值
def getMaxInformationEntropy(gray):
    p_arr = getPixelProbability(gray)
    IES = np.zeros([256, ])
    for t in range(1, len(IES)):
        f_arr, b_arr = separateForegroundAndBackground(p_arr, t)
        IES[t] = informationEntropy(f_arr) + informationEntropy(b_arr)
    return np.argmax(IES)


# 读取图片
image = cv2.imread('lena.png')
# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 获取阈值
t1 = time.time()
T = getMaxInformationEntropy(gray)
print("using time", time.time()-t1)

ret, thresh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
# 可视化 灰度图
cv2.imshow("gray image", gray)
# 可视化 阈值分割图
cv2.imshow("threshold image", thresh)
cv2.waitKey()

