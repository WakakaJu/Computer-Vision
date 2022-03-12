import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

tic = time.time()
I = cv2.imread('fog_road.jpg', 1)
k = 8  # 将照片分割块
height, width, depth = I.shape
ycrcb = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)  # 将RGB图像转入YCbCr空间中
Y = ycrcb[:, :, 0]
Cb = ycrcb[:, :, 1].astype(np.uint8)
Cr = ycrcb[:, :, 2].astype(np.uint8)
# m，n分别为每个小方块的长宽（除最后一行和最后一列）
m = (height // k)
n = (width // k)
Y = Y.astype(np.uint8)


# 每个小块的直方图均衡函数HE
def AutoHE(Y2,m,n):
    # c为各亮度出现次数，总和C为全部像素点数
    c = np.zeros((256, 1))
    for i in range(256):
        c[i] = np.sum(Y2 == i)
    C = np.sum(c)
    # p为各亮度出现概率，总和P为1
    p = c / C
    ck = np.cumsum(p)
    fk = 255 * ck
    fk = np.round(fk)

    Y3 = np.zeros((m * n, 1))
    for j in range(256):
        Y3[Y2 == j] = fk[j]
    return Y3

YN = np.zeros_like(Y)
num_hori = width-n+1  # 水平滑动次数
num_vert = height-m+1  # 竖直滑动次数
print('Total times is',num_hori*num_vert)
num = 0
for i in range(num_hori*num_vert):
    j = 0
    row = i // num_hori
    col = i % num_hori
    Y2 = np.zeros(m * n)
    for j in range(m * n):
        Y2[j] = Y[row + (j // n), col + (j % n)]
    Y3 = AutoHE(Y2,m,n)
    for q in range(m * n):
        YN[row + (q // n), col + (q % n)] = Y3[q]
    num += 1
    print(num)

YN = YN.astype(np.uint8)
ycrcb_out = cv2.merge([YN, Cb, Cr])
# YM = cv2.cvtColor(YN,cv2.COLOR_GRAY2RGB)  # 三通道都是一样的值
# YM = cv2.applyColorMap(YN, cv2.COLORMAP_HSV)  # 伪彩色
G = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCR_CB2BGR)
hist = cv2.calcHist([YN], [0], None, [256], [0, 255])
plt.plot(hist, 'k')
plt.show()
toc = time.time()
print('using time is',toc-tic)
# cv2.imshow('Light', Y)
cv2.imshow('Result', G)
cv2.waitKey(0)
cv2.destroyAllWindows()
