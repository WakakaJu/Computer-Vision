import numpy as np
import cv2
import time

tic = time.time()
I = cv2.imread('4.jpg', 1)
# 彩色图像先分开成BGR三通道，分别进行SSR，再合起来
b,g,r = cv2.split(I)
m = np.size(I, 0)  # m为行数
n = np.size(I, 1)  # n为列数

# 图像预处理（除去0）
def replaceZeros(img):
    min_img = np.min(img[np.nonzero(img)])
    img[img == 0] = min_img
    return img

'''
def SSR(img, size):  # 单尺度Retinex(Single Scale Retinex)  cv2库函数版
    L_blur = cv2.GaussianBlur(img, (size, size), 0)
    h, w = img.shape[:2]
    dst_img = np.zeros((h, w), dtype=np.float32)
    dst_Lblur = np.zeros((h, w), dtype=np.float32)
    dst_R = np.zeros((h, w), dtype=np.float32)

    img = replaceZeros(img).astype(np.float32)
    L_blur = replaceZeros(L_blur).astype(np.float32)
    dst_img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_img, dst_Lblur)
    log_R = cv2.subtract(dst_img, dst_IxL)
    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
'''
def SSR(img,GaussLB_size):  # 单尺度Retinex(Single Scale Retinex)  numpy库函数版
    L_blur = cv2.GaussianBlur(img, (GaussLB_size, GaussLB_size), 0)
    img = replaceZeros(img).astype(np.float32)
    L_blur = replaceZeros(L_blur).astype(np.float32)
    log_R = (np.log(img/255.0) - np.log(img/255.0)*np.log(L_blur/255.0)).astype(np.float32)
    R = (log_R-np.min(log_R))/(np.max(log_R)-np.min(log_R))*255.0
    R = R.astype(np.uint8)
    return R

b_out = SSR(b,3)
g_out = SSR(g,3)
r_out = SSR(r,3)
result = cv2.merge([b_out,g_out,r_out])
toc = time.time()
print('Using time is',toc-tic)
cv2.imshow('Original',I)
cv2.imshow('Result', result)
cv2.imwrite('SSR.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()