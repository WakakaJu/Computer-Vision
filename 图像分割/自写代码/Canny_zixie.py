import cv2
import numpy as np
import time

tic = time.time()
# 读取图片
I = cv2.imread('lena.png', 1)
# 转为灰度图
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# G = G.astype(np.int64)
m = np.size(G, 0)  # m为行数
n = np.size(G, 1)  # n为列数

# 高斯滤波降噪
G = cv2.GaussianBlur(G, (3, 3), 1.5)

# 初始化
G3 = np.zeros((m, n)).astype(np.uint16)  # 梯度图初始化
G4 = np.zeros((m, n)).astype(np.uint16)  # 非最大抑制边缘图初始化
D = np.zeros((m, n)).astype(np.float16)  # 边缘方向（弧度制）初始化
angle = np.zeros((m, n)).astype(np.float16)  # 边缘方向（角度制）初始化

# Sobel算子
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


# 将卷积算子旋转180°
def rotation(a):
    b = np.flipud(a)
    b = np.fliplr(b)
    return b


# 用于和矩阵相乘计算的矩阵
ssx = rotation(sx)
ssy = rotation(sy)


# 对图像卷积计算
def convolution(g, ssx, ssy):
    gx = -np.trace(np.dot(g, ssx))  # x方向导数
    gy = -np.trace(np.dot(ssy, g))  # y方向导数
    # gg = np.hypot(gx,gy)  # 计算欧几里得范数 =sqrt(gx*gx + gy*gy)
    gg = np.abs(gx) + np.abs(gy)  # 欧几里得范数可近似为二者绝对值之和
    gg = gg.astype(np.int64)
    theta = np.arctan2(gy, gx)  # 边缘方向矩阵3×3
    return gg, theta, gx, gy


# 初始化
ggx = np.zeros((m, n))  # x方向梯度
ggy = np.zeros((m, n))  # y方向梯度
g = np.zeros((m, n))

for i in range(1, m - 1):
    for j in range(1, n - 1):
        g = np.array([[G[i - 1, j - 1], G[i - 1, j], G[i - 1, j + 1]], [G[i, j - 1], G[i, j], G[i, j + 1]],
                      [G[i + 1, j - 1], G[i + 1, j], G[i + 1, j + 1]]])
        gg, theta, gx, gy = convolution(g, ssx, ssy)
        G3[i, j] = gg
        D[i, j] = theta
        ggx[i, j] = gx
        ggy[i, j] = gy

angle = D * 180. / np.pi  # 边缘方向角度制
angle[angle < 0] += 180
max_in_line = 255  # 边缘方向所在线上最大值初始化
value_ij = 0  # [i,j]位置上的当前值初始化
p = 0
q = 0

# 非极大抑制
for i in range(1, m - 1):
    for j in range(1, n - 1):
        if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:  # 0度线
            p = G3[i, j - 1]
            q = G3[i, j + 1]
        elif 22.5 <= angle[i, j] < 67.5:  # 45度线
            p = G3[i - 1, j - 1]
            q = G3[i + 1, j + 1]
        elif 67.5 <= angle[i, j] < 112.5:  # 90度线
            p = G3[i - 1, j]
            q = G3[i + 1, j]
        elif 112.5 <= angle[i, j] < 157.5:  # 135度线
            p = G3[i - 1, j + 1]
            q = G3[i + 1, j - 1]
        if (G3[i, j] >= p) and (G3[i, j] >= q):
            G4[i, j] = G3[i, j]
'''
# 插值非极大抑制
weight = np.int64(0)
dTemp1 = 0
dTemp2 = 0
for i in range(1,m-1):
    for j in range(1,n-1):
        if (np.abs(ggy[i, j]) > np.abs(ggx[i, j])) and (ggy[i,j]!=0) and (ggx[i,j]!=0):  # y方向梯度更大
            weight=ggx[i, j]/ggy[i,j]
            g2=G3[i+1,j]
            g4=G3[i-1,j]
            if ggy[i, j] * ggx[i, j] > 0:  # y梯度与x梯度同向
                g1 = G3[i + 1, j - 1]
                g3 = G3[i - 1, j + 1]
            elif ggy[i, j] * ggx[i, j] < 0:  # y梯度与x梯度反向
                g1 = G3[i + 1, j + 1]
                g3 = G3[i - 1, j - 1]
        elif (np.abs(ggy[i, j]) < np.abs(ggx[i, j]))  and (ggy[i,j]!=0) and (ggx[i,j]!=0):  # x方向梯度更大
            weight = ggy[i, j] / ggx[i, j]
            g2 = G3[i, j - 1]
            g4 = G3[i, j + 1]
            if ggy[i, j] * ggx[i, j] > 0:
                g1 = G3[i - 1, j - 1]
                g3 = G3[i + 1, j + 1]
            elif ggy[i, j] * ggx[i, j] < 0:
                g1 = G3[i + 1, j - 1]
                g3 = G3[i - 1, j + 1]
        dTemp1 = weight * g1 + (1.0 - weight) * g2
        dTemp2 = weight * g3 + (1.0 - weight) * g4
        if G3[i,j] >= 100 and G3[i, j] >= 25:
             G4[i,j] = G3[i,j]
'''
highThreshold = 100  # 高阈值
lowThreshold = 80  # 低阈值
strong_i, strong_j = np.where(G4 > highThreshold)
zeros_i, zeros_j = np.where(G4 < lowThreshold)
weak_i, weak_j = np.where((G4 < highThreshold) & (G4 >= lowThreshold))

G5 = np.zeros((m,n))
G5[strong_i,strong_j] = 255
G5[weak_i,weak_j] = 50

# G4 = G4.astype(np.uint8)
G3 = G3.astype(np.uint8)
G = G.astype(np.uint8)

toc = time.time()
print('using time is ',toc-tic)

cv2.imshow('GrayImage', G)
cv2.imshow('GradientMap', G3)
cv2.imshow('Edge', G5)
cv2.waitKey(0)

cv2.destroyAllWindows()
