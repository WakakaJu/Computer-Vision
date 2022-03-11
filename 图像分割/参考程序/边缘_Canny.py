import cv2
import time

tic = time.time()
img = cv2.imread('lena.png',0)  # 其中，0表示将图片以灰度读出来。

img = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。
canny = cv2.Canny(img, 25, 100)  # 最大最小阈值

toc = time.time()
print('using time is',toc-tic)
cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
