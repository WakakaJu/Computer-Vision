import numpy as np
import cv2
import time

tic = time.time()
I = cv2.imread('4.jpg', 1)
# 彩色图像先分开成BGR三通道，分别进行SSR，再合起来
b, g, r = cv2.split(I)
m = np.size(I, 0)  # m为行数
n = np.size(I, 1)  # n为列数

# 图像预处理（除去0）
def replaceZeros(img):
    min_img = np.min(img[np.nonzero(img)])
    img[img == 0] = min_img
    return img

'''
# 多尺度Retinex(Multi Scale Retinex)  cv2库函数版
def MSR(img, GaussLB_scales):
    num_scales = len(GaussLB_scales)
    weight = 1/num_scales
    h, w = img.shape[:2]
    img = replaceZeros(img).astype(np.float32)
    log_R = np.zeros((h,w),dtype=np.float32)

    for i in range(num_scales):
        L_blur = cv2.GaussianBlur(img, (GaussLB_scales[i], GaussLB_scales[i]), 0)
        L_blur = replaceZeros(L_blur).astype(np.float32)
        dst_img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_IxL = cv2.multiply(dst_img, dst_Lblur)
        log_R += weight*cv2.subtract(dst_img, dst_IxL)
    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
'''

# 多尺度Retinex(Multi Scale Retinex)  numpy库函数版
# img为一层二维数组，L_blur为高斯滤波尺度数层个二维数组
def MSR(img, GaussLB_scales):
    num_scales = len(GaussLB_scales)
    weight = 1 / num_scales
    img = replaceZeros(img).astype(np.float32)
    height, width = img.shape
    L_blur = np.zeros((height, width), dtype=np.float32)
    log_R = np.zeros((height, width), dtype=np.float32)
    for i in range(num_scales):
        L_blur = cv2.GaussianBlur(img, (GaussLB_scales[i], GaussLB_scales[i]), 0)
        L_blur = replaceZeros(L_blur).astype(np.float32)
        log_R += weight * (np.log(img / 255.0) - np.log(img / 255.0) * np.log(L_blur / 255.0))
    R = (log_R - np.min(log_R)) / (np.max(log_R) - np.min(log_R)) * 255.0
    R = R.astype(np.uint8)
    return R

GaussLB_scales = [3, 5, 15]
b_out = MSR(b, GaussLB_scales)
g_out = MSR(g, GaussLB_scales)
r_out = MSR(r, GaussLB_scales)
result = cv2.merge([b_out, g_out, r_out])
toc = time.time()
print('Using time is', toc - tic)
cv2.imshow('Original', I)
cv2.imshow('Result', result)
cv2.imwrite('MSR.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()