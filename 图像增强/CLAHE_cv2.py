import cv2
from matplotlib import pyplot as plt
import time

tic = time.time()
image = cv2.imread('fog_road.jpg', cv2.IMREAD_COLOR)
b, g, r = cv2.split(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
image = cv2.merge([b, g, r])
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)  # 将RGB图像转入YCbCr空间中
Y = ycrcb[:, :, 0]
hist = cv2.calcHist([Y], [0], None, [256], [0, 255])
toc = time.time()
print('Using time is', toc - tic)
plt.plot(hist, 'k')
plt.show()
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
