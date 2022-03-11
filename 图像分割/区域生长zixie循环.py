import numpy as np
import cv2

I = cv2.imread('lena.png',1)
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
m = np.size(G,0) # m为行数
n = np.size(G,1) # n为列数
T = 10  # 门限
area_num = 1  # 区域个数
seed_i = np.zeros(1)
seed_j = np.zeros(1)
# for i in range(area_num):
#     location[i,]

seed_i[0] = np.random.randint(0,m)
seed_j[0] = np.random.randint(0,n)

result = np.zeros((m,n))
dashu = 10000
k = 0
for i in range(0, m):
    for j in range(0, n):
        c = np.array([[G[i - 1, j - 1], G[i - 1, j], G[i - 1, j + 1]],[G[i, j - 1], 0, G[i, j + 1]],[G[i + 1, j - 1], G[i + 1, j], G[i + 1, j + 1]]])
while k<dashu:
    c = np.array([[G[seed_i[k] - 1, seed_j[k] - 1], G[seed_i[k] - 1, seed_j[k]], G[seed_i[k] - 1, seed_j[k] + 1]], [G[seed_i[k], seed_j[k] - 1], 0, G[seed_i[k], seed_j[k] + 1]],
                  [G[seed_i[k] + 1, seed_j[k] - 1], G[seed_i[k] + 1, seed_j[k]], G[seed_i[k] + 1, seed_j[k] + 1]]])



