import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

tic = time.time()
I = cv2.imread('fog_road.jpg', 1)
k = 8  # 将照片分割块
threshold = 100  # 累计直方图函数CDF阈值（某些亮度的像素个数）
height, width, depth = I.shape
I = I[0:(height - (height % k)), 0:(width - (width % k)), :]
ycrcb = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)  # 将RGB图像转入YCbCr空间中
# channel = cv2.split(ycrcb)  # 将YCbCr通道分离
Y = ycrcb[:, :, 0]
Cb = ycrcb[:, :, 1].astype(np.uint8)
Cr = ycrcb[:, :, 2].astype(np.uint8)
height1, width1 = Y.shape
# 将亮度图转为一列一维数组
Y2 = Y.reshape(height1 * width1, 1)
# m，n分别为每个小方块的长宽（除最后一行和最后一列）
m = (height1 // k)
n = (width1 // k)
Y = Y.astype(np.uint8)


# 每个小块的直方图均衡函数HE
def AutoHE(Y2):
    # c为各亮度出现次数，总和C为全部像素点数
    c = np.zeros((256, 1))
    for i in range(256):
        c[i] = np.sum(Y2 == i)

    # 修剪直方图，限制对比度
    cp = c - threshold
    cp[cp < 0] = 0
    c[c > threshold] = threshold
    c = c + np.sum(cp) / 256

    C = np.sum(c)
    # p为各亮度出现概率，总和P为1
    p = c / C
    ck = np.cumsum(p)
    fk = 255 * ck
    fk = np.round(fk)

    Y3 = np.zeros((m * n, 1))
    for j in range(256):
        Y3[Y2 == j] = fk[j]
    return Y3, fk


# 嵌套列表重组函数
def list_chongzu(row_maps, num_lists):
    maps = []
    num_array = len(row_maps) // num_lists
    i = 0
    for j in range(num_lists):
        pp = []
        for jj in range(num_array):
            p = row_maps[i]
            pp.append(p)
            i += 1
        maps.append(pp)
    return maps


row_maps = []
YN = np.zeros_like(Y)
for i in range(k ** 2):
    j = 0
    row = (i // k) * m
    col = (i % k) * n

    Y22 = np.zeros(m * n)
    for j in range(m * n):
        Y22[j] = Y[row + (j // n), col + (j % n)]
    Y3, fk = AutoHE(Y22)
    row_maps.append(fk)
    for q in range(m * n):
        YN[row + (q // n), col + (q % n)] = Y3[q]
YN = YN.astype(np.uint8)
maps = list_chongzu(row_maps, num_lists=8)

YK = Y.copy()
for i in range(height1):
    for j in range(width1):
        r = int((i - m / 2) / m)  # 左上映射函数的行索引
        c = int((j - n / 2) / n)  # 左上映射函数的列索引

        x1 = (i - (r + 0.5) * m) / m  # 到左上映射中心的x轴距离
        y1 = (j - (c + 0.5) * n) / n  # 到左上映射中心的y轴距离

        lu = 0  # 左上CDF映射值
        lb = 0  # 左下
        ru = 0  # 右上
        rb = 0  # 右下

        # 四角直接使用最近的映射
        if r < 0 and c < 0:
            YK[i][j] = maps[r + 1][c + 1][Y[i][j]]
        elif r < 0 and c >= k - 1:
            YK[i][j] = maps[r + 1][c][Y[i][j]]
        elif r >= k - 1 and c < 0:
            YK[i][j] = maps[r][c + 1][Y[i][j]]
        elif r >= k - 1 and c >= k - 1:
            YK[i][j] = maps[r][c][Y[i][j]]
        # 四周情况使用相邻两个区域线性插值
        elif r < 0 or r >= k - 1:
            if r < 0:
                r = 0
            elif r > k - 1:
                r = k - 1
            left = maps[r][c][Y[i][j]]
            right = maps[r][c + 1][Y[i][j]]
            YK[i][j] = (1 - y1) * left + y1 * right
        elif c < 0 or c >= k - 1:
            if c < 0:
                c = 0
            elif c > k - 1:
                c = k - 1
            up = maps[r][c][Y[i][j]]
            bottom = maps[r + 1][c][Y[i][j]]
            YK[i][j] = (1 - x1) * up + x1 * bottom
        # 内部像素使用双线性插值
        else:
            lu = maps[r][c][Y[i][j]]
            lb = maps[r + 1][c][Y[i][j]]
            ru = maps[r][c + 1][Y[i][j]]
            rb = maps[r + 1][c + 1][Y[i][j]]
            YK[i][j] = (1 - y1) * ((1 - x1) * lu + x1 * lb) + y1 * ((1 - x1) * ru + x1 * rb)

ycrcb_out = cv2.merge([YK, Cb, Cr])
G = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCR_CB2BGR)
hist = cv2.calcHist([YK], [0], None, [256], [0, 255])
toc = time.time()
print('Using time is', toc - tic)
plt.plot(hist, 'k')
plt.show()
cv2.imshow('Light', Y)
cv2.imshow('Original', I)
cv2.imshow('Result', G)
cv2.waitKey(0)
cv2.destroyAllWindows()

