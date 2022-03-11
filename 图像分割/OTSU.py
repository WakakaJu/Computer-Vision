# 导入库
import cv2
import time

# 读取图片
image = cv2.imread('lena.png',1)
# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 调取opencv库自带的OTSU方法
t1 = time.time()
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
print("using time", time.time()-t1)
# 可视化 灰度图
cv2.imshow("gray image", gray)
# 可视化 阈值分割图
cv2.imshow("threshold image", thresh)
cv2.waitKey()

ddghet