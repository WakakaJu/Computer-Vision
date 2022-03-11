import numpy as np
import cv2
import time
'''
由于马尔可夫随机场是先设定一个随机数组，再在每次迭代过程中使其朝图像发展，
因此不能确定图像分类的具体情况，即
可能出现相同的图像在重新运行程序时本来的第一类区域变成第二类，
本来的第二类区域变成第一类的（即反色）情况，属正常现象
'''
tic = time.time()

I = cv2.imread('lena.png', 1)
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
m = np.size(G, 0)  # m为行数
n = np.size(G, 1)  # n为列数
T = 100  # 初始温度 Gibbs随机场的概率P公式内部参数
# 设置分类数
cluster_sum = 4
# 设置最大循环次数
maxiter = 10
# 初始化
label = np.zeros((m, n))
label = np.random.randint(1, cluster_sum + 1, [m, n])
ps = G


# 定义函数：马尔可夫随机场
def markov_random_fields(label, ps):  # label为最初随机标签，ps为S概率，kk为当时循环次数
    # 函数内部初始化
    label_m = np.zeros((m + 2, n + 2)).astype(np.int64)
    label_m[1:m + 1, 1:n + 1] = label[0:m, 0:n]
    z = np.zeros((m, n))
    uw = np.zeros((m, n, cluster_sum))
    pw = np.zeros((m, n, cluster_sum))
    c = np.zeros((3, 3))

    label_k = np.zeros((m, n))
    gs_mean = np.zeros(cluster_sum)
    gs_sigma = np.zeros(cluster_sum)
    gs_lamda = np.zeros(cluster_sum)

    psw = np.zeros((m, n, cluster_sum))
    pws = np.zeros((m, n, cluster_sum))  # 目标先验概率

    beta = 0.8  # beta为耦合系数，通常为0.5-1
    for i in range(0, m):
        for j in range(0, n):
            c = np.array([[label_m[i - 1, j - 1], label_m[i - 1, j], label_m[i - 1, j + 1]],
                          [label_m[i, j - 1], 0, label_m[i, j + 1]],
                          [label_m[i + 1, j - 1], label_m[i + 1, j], label_m[i + 1, j + 1]]])
            for k in range(0, cluster_sum):
                uw[i, j, k] = beta * (8 - 2 * np.sum(c == (k + 1)))

    uuw = np.exp(-1 * uw / T)
    z = np.sum(uuw, axis=2)

    for k in range(0, cluster_sum):
        pw[:, :, k] = 1 / z * uuw[:, :, k]
        label_i, label_j = np.where(label[:, :] == (k + 1))
        label_k = ps[label_i, label_j]
        gs_mean[k] = np.mean(label_k)
        gs_sigma[k] = np.var(label_k)
        gs_lamda[k] = 1 / np.sqrt(2 * np.pi * gs_sigma[k])  # 高斯密度函数 系数
        psw[:, :, k] = gs_lamda[k] * np.exp(-(ps - gs_mean[k]) * (ps - gs_mean[k]) / 2 / gs_sigma[k])
        pws[:, :, k] = psw[:, :, k] * pw[:, :, k] / ps

    label[:, :] = np.argmax(pws, axis=2) + 1
    return label


kk = 0
while kk < maxiter - 1:
    label = markov_random_fields(label, ps)
    kk += 1
    print(kk)

# final  =  maxiter - 1
area_result = np.zeros((m, n)).astype(np.uint8)
area_result[label[:, :] == 1] = 0
area_result[label[:, :] == 2] = 255
area_result[label[:, :] == 3] = 85
area_result[label[:, :] == 4] = 170

toc = time.time()
print('time is ', toc - tic)

cv2.imshow('area_result', area_result)
cv2.waitKey(0)

cv2.destroyAllWindows()
