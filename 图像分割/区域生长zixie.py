import numpy as np
import cv2

I = cv2.imread('lena.png',1)
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
m = np.size(G,0) # m为行数
n = np.size(G,1) # n为列数
T = 10  # 门限
area_num = 1  # 区域个数
seed_location = np.zeros((2,area_num))
# for i in range(area_num):
#     location[i,]

seed_location[0,0] = np.random.randint(0,m)
seed_location[0,1] = np.random.randint(0,n)

result = np.zeros((m,n))

f_u = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_d = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)
f_l = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3)
f_r = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(3, 3)
f_ul = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_ur = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
f_dl = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3)
f_dr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(3, 3)

G_u = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_u)
G_d = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_d)
G_l = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_l)
G_r = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_r)
G_ul = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_ul)
G_ur = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_ur)
G_dl = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_dl)
G_dr = cv2.filter2D(np.array(G, dtype=np.uint8), -1, f_dr)



