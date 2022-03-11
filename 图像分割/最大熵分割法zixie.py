import cv2
import numpy as np


# 读取图片
I = cv2.imread('lena.png',1)
# 转为灰度图
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
m = np.size(G,0) # m为行数
n = np.size(G,1) # n为列数
# 将灰度图转为一列一维数组
G2 = G.reshape(m*n,1)
mmm = np.argmin(G2) # 测试：最小灰度值为2
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
k = 80
kk = np.arange(1,256).reshape(255,1)
phi = np.zeros((255,1))

# 定义函数maxphi
def maxphi(k,c,p):
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
    h1 = (p1/P1)*np.log(p1/P1)
    h2 = (p2/P22)*np.log(p2/P22)
    h1[np.isnan(h1)] = 0
    h2[np.isnan(h2)] = 0
    # H1,H2是分割的两个图像的熵
    H1 = np.sum(h1)
    H2 = np.sum(h2)
    # 计算熵和phi
    phi = -H1 - H2
    return phi

for k in range(1,254):
    phi[k] = maxphi(k,c,p)
phi[np.isnan(phi)] = 0
T = np.argmax(phi)+1 # T为最大类间方差时的灰度值

# 绘制分割图像
GG1 = np.zeros((m*n,1))
GG1 = 255 - GG1
GG2 = np.zeros((m*n,1))
GG2 = 255 - GG2
for s in range(0,m*n):
    if G2[s] <= T:
        GG1[s] = 0  # G2[s]
    else:
        GG2[s] = 0  # G2[s]
GG1 = GG1.reshape(m,n)
GG2 = GG2.reshape(m,n)

'''
此时GG1,GG2都是浮点型255.0，
如果直接用cv2.imshow画图则显示不出来（当做小数，不在0-255范围内），
需要将其转化为整型，用unit8显示最好（白色）
'''
GG1 = GG1.astype(np.uint8)
GG2 = GG2.astype(np.uint8)

cv2.resizeWindow('dark image', m, n)
cv2.resizeWindow('light image', m, n)
cv2.imshow('setting',GG1)
cv2.imshow('image',GG2)
cv2.waitKey(0)

cv2.destroyAllWindows()
