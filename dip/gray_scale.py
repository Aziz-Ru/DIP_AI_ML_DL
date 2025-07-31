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
def build_grayScale():
    
    imgpath1 = os.path.join(os.getcwd(), 'image/paddy_image.jpeg')
    img = cv.imread(imgpath1, 1)
    grayimg = cv.imread(imgpath1, 0)
    rgbImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    rows,cols,channels=rgbImg.shape

    red=rgbImg[:,:,0]
    green=rgbImg[:,:,1]
    blue=rgbImg[:,:,2]
    for i in range(rows):
        for j in range(cols):
            red[i,j]=red[i,j]*0.299

    for i in range(rows):
        for j in range(cols):
            green[i,j]=green[i,j]*0.114
    
    for i in range(rows):
        for j in range(cols):
            blue[i,j]=blue[i,j]*0.587

    plt.subplot(3,2,1)
    plt.imshow(img)
    
    plt.subplot(3,2,2)
    plt.imshow(bgr_to_rgb(img))

    plt.subplot(3,2,3)
    plt.imshow(rgbImg)
    
    

    # for i in range(rows):
    #     for j in range(cols):
    #         red[i,j]=red[i,j]/3

    # for i in range(rows):
    #     for j in range(cols):
    #         green[i,j]=green[i,j]/3
    
    # for i in range(rows):
    #     for j in range(cols):
    #         blue[i,j]=blue[i,j]/3

    # plt.subplot(3,2,6)
    # plt.imshow(red+green+blue,cmap='gray')
    
    

    
    plt.show()


def bgr_to_rgb(img):
    return img[:,:,::-1]