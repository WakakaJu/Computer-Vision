import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

tic = time.time()
I = cv2.imread('fog_road.jpg', 1)
ycrcb = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
Y = ycrcb[:,:,0]
Cb = ycrcb[:,:,1].astype(np.uint8)
Cr = ycrcb[:,:,2].astype(np.uint8)
m = np.size(Y, 0)  # m为行数
n = np.size(Y, 1)  # n为列数
# 将灰度图转为一列一维数组
Y2 = Y.reshape(m * n, 1)
Y2 = Y2.astype(np.int32)
mean_gray = (np.max(Y2) + np.min(Y2)) / 2  # 亮度均值
int_mean = int(mean_gray)
M1 = np.zeros_like(Y2)  # M1,M2为由亮度均值切出来的两幅子图
M2 = np.zeros_like(Y2)
M1[Y2 <= int_mean] = Y2[Y2 <= int_mean]
M2[Y2 > int_mean] = Y2[Y2 > int_mean]
# n为各灰度出现次数，总和N为全部像素点数
c = 0
c1 = np.zeros((int_mean + 1, 1))
c2 = np.zeros((255 - int_mean, 1))
c = np.sum(Y2 == 0)
for i in range(1, 256):
    if i < int_mean + 1:
        c1[i] = np.sum(M1 == i)
    else:
        c2[i - int_mean - 1] = np.sum(M2 == i)
c1[0] = c
C1 = np.sum(c1)
C2 = np.sum(c2)
# p为各灰度出现概率，总和P为1
p1 = c1 / C1
p2 = c2 / C2
ck1 = np.cumsum(p1)  # ck为累计概率
ck2 = np.cumsum(p2)
fk1 = int_mean * ck1
fk2 = (255 - int_mean - 1) * ck2 + int_mean + 1
fk1 = np.round(fk1)
fk2 = np.round(fk2)
Y = Y.astype(np.uint8)
MM1 = np.zeros((m * n, 1))
MM2 = np.zeros((m * n, 1))
for j in range(0, 256):
    if j < int_mean + 1:
        MM1[M1 == j] = fk1[j]
    else:
        MM2[M2 == j] = fk2[j - int_mean - 1]
Y3 = MM1 + MM2
YN = Y3.reshape(m, n)
YN = YN.astype(np.uint8)
ycrcb_out = cv2.merge([YN,Cb,Cr])
G = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCR_CB2BGR)
hist = cv2.calcHist([YN], [0], None, [256], [0, 255])
plt.plot(hist,'k')
plt.show()
toc=time.time()
print('Using time is',toc-tic)
# cv2.imshow('Light', Y)
cv2.imshow('Result', G)
cv2.waitKey(0)
cv2.destroyAllWindows()

