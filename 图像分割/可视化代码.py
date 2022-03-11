import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


# 读取图片
image1 = cv2.imread("../data/dog.jpg")
image2 = cv2.imread("../data/dog2.jpg")
image3 = cv2.imread("../data/cat1.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

# 转为灰度图
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
# # OSTU
# ret1, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
# ret2, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
# ret3, thresh3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_OTSU)

# 最大熵阈值
# 获取各个像素的概率
# def getPixelProbability(gray):
#     p_arr = np.zeros([256, ])
#     gray = gray.ravel()
#     for p in gray:
#         p_arr[p] += 1
#     return p_arr / len(gray)
#
#
# # 分离前景和背景
# def separateForegroundAndBackground(arr, t):
#     f_arr, b_arr = arr[:t], arr[t:]
#     return f_arr, b_arr
#
#
# # 获取某阈值下的信息熵
# def informationEntropy(arr):
#     Pn = np.sum(arr)
#     if Pn == 0:
#         return 0
#     iE = 0
#     for p_i in arr:
#         if p_i == 0:
#             continue
#         iE += (p_i / Pn) * np.log(p_i / Pn)
#     return -iE
#
#
# # 获取最优阈值
# def getMaxInformationEntropy(gray):
#     p_arr = getPixelProbability(gray)
#     IES = np.zeros([256, ])
#     for t in range(1, len(IES)):
#         f_arr, b_arr = separateForegroundAndBackground(p_arr, t)
#         IES[t] = informationEntropy(f_arr) + informationEntropy(b_arr)
#     return np.argmax(IES)
#
#
# T1 = getMaxInformationEntropy(gray1)
# T2 = getMaxInformationEntropy(gray2)
# T3 = getMaxInformationEntropy(gray3)
# ret1, thresh1 = cv2.threshold(gray1, T1, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(gray2, T2, 255, cv2.THRESH_BINARY)
# ret3, thresh3 = cv2.threshold(gray3, T3, 255, cv2.THRESH_BINARY)

# 迭代阈值
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


T1 = getOptimalThreshold(gray1)
T2 = getOptimalThreshold(gray2)
T3 = getOptimalThreshold(gray3)
ret1, thresh1 = cv2.threshold(gray1, T1, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(gray2, T2, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(gray3, T3, 255, cv2.THRESH_BINARY)

# 可视化
figure, axes = plt.subplots(2, 3, figsize=(9, 6))
axes[0][0].set_title('original image A')
axes[0][0].imshow(image1)
axes[0][1].set_title('original image B')
axes[0][1].imshow(image2)
axes[0][2].set_title('original image C')
axes[0][2].imshow(image3)
axes[1][0].set_title('processed image A')
axes[1][0].imshow(thresh1, cmap='gray')
axes[1][1].set_title('processed image B')
axes[1][1].imshow(thresh2, cmap='gray')
axes[1][2].set_title('processed image C')
axes[1][2].imshow(thresh3, cmap='gray')

plt.show()



