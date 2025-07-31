import cv2 as cv 
import matplotlib.pyplot as plt 
import os
import numpy as np
import math
# s = c* r^g
# s= c * log2(1+r)

# red changell 0.299
# blue channel 0.587
# green channe 0.114
def build_nonlinear():
    print(os.getcwd())
    imgpath1 = os.path.join(os.getcwd(), 'image/paddy_image.jpeg')
    img = cv.imread(imgpath1, 0)
  
    def make_gamma(gamma):
        no_rows,no_col=img.shape
        new_img=np.zeros((no_rows,no_col),dtype=np.uint8)
        # for channel in range(no_channel):
        for row in range (no_rows):
            for col in range (no_col):
                new_img[row][col]=img[row][col]**gamma

        return new_img 
    def make_log():
        no_rows,no_col=img.shape
        new_img=np.zeros((no_rows,no_col),dtype=np.uint8)
        # for channel in range(no_channel):
        for row in range (no_rows):
            for col in range (no_col):
                value = math.log2(1 + float(img[row, col])) 
                new_img[row,col]=int(value)
                # print(value)

        return new_img 
# 0.1,0.3,0.7,1,2,3
    gamma_img1=make_gamma(0.1)
    gamma_img2=make_gamma(0.3)
    gamma_img3=make_gamma(0.7)
    gamma_img4=make_gamma(1)
    gamma_img5=make_gamma(2)
    log_img=make_log()
    plt.figure(figsize=(10,10))
    plt.subplot(3,3,1)
    plt.imshow(img,cmap='grey')
    plt.subplot(3,3,2)
    plt.imshow(gamma_img1,cmap='grey')

    plt.subplot(3,3,3)
    plt.imshow(gamma_img2,cmap='grey')

    plt.subplot(3,3,4)
    plt.imshow(gamma_img3,cmap='grey')

    plt.subplot(3,3,5)
    plt.imshow(gamma_img4,cmap='grey')

    plt.subplot(3,3,5)
    plt.imshow(gamma_img5,cmap='grey')

    plt.subplot(3,3,6)
    plt.imshow(log_img,cmap='grey')

    plt.show()

