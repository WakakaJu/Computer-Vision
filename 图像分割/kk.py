# 导入库
import cv2
import numpy as np


img = cv2.imread("../data/apple.jpg")
h, w, _ = img.shape
img = cv2.resize(img, (2*h, w*2))
cv2.imwrite("../data/800p.jpg", img)

