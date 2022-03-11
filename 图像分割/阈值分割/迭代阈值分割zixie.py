import cv2
import numpy as np

# 读取图片
I = cv2.imread('lena.png',1)
# 转为灰度图
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
m = np.size(G,0)  # m为行数
n = np.size(G,1)  # n为列数
# 将灰度图转为一列一维数组
G2 = G.reshape(m*n,1)
mmm = np.argmin(G2)  # 测试：最小灰度值为2

# 定义函数：迭代阈值
# def iterationFindT(G2):
zmax = np.max(G2).astype(np.uint16)  # 最大灰度值
zmin = np.min(G2).astype(np.uint16)  # 最小灰度值

T = np.zeros((m*n,1))  # 定义阈值序列
z = np.zeros((m*n,2)).astype(np.uint16)
z1 = z[:,0]
z2 = z[:,1]
z[0,0] = zmax
z[0,1] = zmin
T[0] = (zmax + zmin)/2  # 初始阈值
k = 0
while k < 200:
    GG1 = np.zeros((m * n, 1))
    GG2 = np.zeros((m * n, 1))
    for i in range(0,m*n):
        if G2[i] > T[k]:
            GG2[i] = 255  # G2[i]
        # else:
        #     GG1[i] = 255  # G2[i]
    '''
    GG1[G2 < T[k]] = 255
    GG1[G2 >= T[k]] = G2
    GG2[G2 < T[k]] = G2
    GG2[G2 >= T[k]] = 255
    '''

    z1[k+1] = (np.sum(GG1))/np.sum(GG1 != 0)
    z2[k+1] = (np.sum(GG2))/np.sum(GG2 != 0)
    T[k+1] = (z1[k+1] + z2[k+1])/2
    if T[k+1] - T[k] <1:
        print(k)
        break
    # else:     # 如果没必要的话，别加else：continue，进入死循环，害人精
    #     continue
    k += 1


# 绘制分割图像
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
