import numpy as np
import cv2
from matplotlib import pyplot as plt

def addsalt_pepper(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)     # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声

    return img_


img = cv2.imread('../data/12.png')

SNR_list = [0.9, 0.7, 0.5, 0.3]
sub_plot = [221, 222, 223, 224]

plt.figure(1)
for i in range(len(SNR_list)):
    plt.subplot(sub_plot[i])
    img_s = addsalt_pepper(img.transpose(2, 1, 0), SNR_list[i])     # c,
    img_s = img_s.transpose(2, 1, 0)
    # cv2.imshow('PepperandSalt', img_s)
    cv2.waitKey(0)
    plt.imshow(img_s[:,:,::-1])     # bgr --> rgb
    plt.title('add salt pepper noise(SNR={})'.format(SNR_list[i]))

plt.show()
