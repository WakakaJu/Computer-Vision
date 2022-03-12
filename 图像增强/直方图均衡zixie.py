import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

tic = time.time()
I = cv2.imread('fog_road.jpg',1)
ycrcb = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)  # 将RGB图像转入YCbCr空间中
# channel = cv2.split(ycrcb)  # 将YCbCr通道分离
Y = ycrcb[:,:,0]
Cb = ycrcb[:,:,1].astype(np.uint8)
Cr = ycrcb[:,:,2].astype(np.uint8)
m = np.size(Y,0) # m为行数
n = np.size(Y,1) # n为列数
# 将灰度图转为一列一维数组
Y2 = Y.reshape(m*n,1)
# c为各灰度出现次数，总和C为全部像素点数
c = np.zeros((256,1))
for i in range(0,256):
    c[i] = np.sum(Y2==i)
# c = c[1:]
C = np.sum(c)
# p为各灰度出现概率，总和P为1
p = c/C
ck = np.cumsum(p)
fk = 255*ck
fk = np.round(fk)
Y = Y.astype(np.uint8)
Y3 = np.zeros((m*n,1))
for j in range(256):
    Y3[Y2==j] = fk[j]
YN= Y3.reshape(m,n)
YN = YN.astype(np.uint8)
ycrcb_out = cv2.merge([YN,Cb,Cr])
# YM = cv2.cvtColor(YN,cv2.COLOR_GRAY2RGB)  # 三通道都是一样的值
# YM = cv2.applyColorMap(YN, cv2.COLORMAP_HSV)  # 伪彩色
G = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCR_CB2BGR)
hist = cv2.calcHist([YN], [0], None, [256], [0, 255])
plt.plot(hist,'k')
plt.show()
toc = time.time()
print('Using time is',toc-tic)
# cv2.imshow('Light',Y)
cv2.imshow('Result',G)
cv2.waitKey(0)
cv2.destroyAllWindows()
