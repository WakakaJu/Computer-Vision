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

G = cv2.imread('lena.png',0)
cluster_num = 2
maxiter = 50
m = np.size(G, 0)  # m为行数
n = np.size(G, 1)  # n为列数

G_double = np.array(G, dtype=np.float64)

label = np.random.randint(1, cluster_num + 1, [m,n])

iter = 0

f_u = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_d = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)
f_l = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3)
f_r = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(3, 3)
f_ul = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_ur = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_dl = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3)
f_dr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(3, 3)

while iter < maxiter:
    iter = iter + 1
    print(iter)

    label_u = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_u)
    label_d = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_d)
    label_l = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_l)
    label_r = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_r)
    label_ul = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ul)
    label_ur = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ur)
    label_dl = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dl)
    label_dr = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dr)
    p_c = np.zeros((cluster_num, m, n))

    for i in range(cluster_num):
        label_i = (i + 1) * np.ones((m, n))
        u_T = 1 * np.logical_not(label_i - label_u)
        d_T = 1 * np.logical_not(label_i - label_d)
        l_T = 1 * np.logical_not(label_i - label_l)
        r_T = 1 * np.logical_not(label_i - label_r)
        ul_T = 1 * np.logical_not(label_i - label_ul)
        ur_T = 1 * np.logical_not(label_i - label_ur)
        dl_T = 1 * np.logical_not(label_i - label_dl)
        dr_T = 1 * np.logical_not(label_i - label_dr)
        temp = u_T + d_T + l_T + r_T + ul_T + ur_T + dl_T + dr_T

        p_c[i, :] = (1.0 / 8) * temp

    p_c[p_c == 0] = 0.001

    mu = np.zeros((1, cluster_num))
    sigma = np.zeros((1, cluster_num))
    for i in range(cluster_num):
        index = np.where(label == (i + 1))
        data_c = G[index]
        mu[0, i] = np.mean(data_c)
        sigma[0, i] = np.var(data_c)
    # p_sc为psw，是w成立条件下s的条件概率
    p_sc = np.zeros((cluster_num, m, n))
    one_a = np.ones((m, n))

    for j in range(cluster_num):
        MU = mu[0, j] * one_a
        p_sc[j, :] = (1.0 / np.sqrt(2 * np.pi * sigma[0, j])) * np.exp(-1. * ((G - MU) ** 2) / (2 * sigma[0, j]))

    X_out = np.log(p_c) + np.log(p_sc)
    label_c = X_out.reshape(cluster_num, m*n)
    label_c_t = label_c.T
    label_m = np.argmax(label_c_t, axis=1)
    label_m = label_m + np.ones(label_m.shape)
    label = label_m.reshape(m, n)

label = label - np.ones(label.shape)  # 为了出现0
label = label.astype(np.uint8)
lable_w = 255 * (1-label)  # 此处做法只能显示两类，一类用0表示另一类用255表示

toc = time.time()
print('time is ',toc-tic)
cv2.imshow('label.jpg', lable_w)
cv2.waitKey(0)

cv2.destroyAllWindows()
