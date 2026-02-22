
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os



def load_image (mode=0):
    dirs = os.getcwd().split(os.sep)
    dirs.pop()
    path = os.sep.join(dirs)+os.sep+'images/tulip.jpeg'
    
    print(path)
    img = cv.imread(path,mode)
    if img is None:
        print("failed to load image")
        exit(1)
    return img


img = load_image()
h , w= img.shape


def step_fun(th):
    new_img = np.zeros(img.shape,dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i,j]>th:
                new_img[i,j]=255
    return new_img


img1= step_fun(90)

plt.figure(figsize=(12,6))
plt.subplot(2,3,1)
plt.imshow(img1,cmap='gray')



plt.show()
