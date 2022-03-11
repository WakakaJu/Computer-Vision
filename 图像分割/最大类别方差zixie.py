import cv2
import numpy as np

# 读取图片
I = cv2.imread('800p.jpg',1)
# 转为灰度图
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
m = np.size(G,0) # m为行数
n = np.size(G,1) # n为列数
# 将灰度图转为一列一维数组
G2 = G.reshape(m*n,1)
# c为各灰度出现次数，总和C为全部像素点数
c = np.zeros((256,1))
a = np.arange(1,256).reshape(255,1)
for i in range(0,256):
    c[i] = np.sum(G2==i)
c = c[1:]
C = np.sum(c)
# p为各灰度出现概率，总和P为1
p = c/C
P = np.sum(p)
# 设定阈值k
k = 255
# 根据设定的阈值分割图像
c1 = np.zeros((255,1))
c2 = np.zeros((255,1))
p1 = np.zeros((255,1))
p2 = np.zeros((255,1))
for i in range(0,255):
    if i <= k:
        c1[i] = c[i]
        c2[i] = 0
        p1[i] = p[i]
        p2[i] = 0
    else:
        c1[i] = 0
        c2[i] = c[i]
        p1[i] = 0
        p2[i] = p[i]
P1 = np.sum(p1)
P2 = np.sum(p2)
P22 = 1-P1 # 检验是否等于P2, 用P22，P2有误差
# m1,m2是分割的两个图像的平均灰度值
m1 = np.sum(a*p1)/P1
m2 = np.sum(a*p2)/P22
mg = np.sum(G2)/m/n
# 计算类间平方差theta
# theta = P1*(m1-mg)*(m1-mg)+P2*(m2-mg)*(m2-mg)
theta = P1*P2*(m1-m2)*(m1-m2)

sdgae