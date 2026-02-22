
import  matplotlib.pyplot as  plt
import cv2 as cv
import numpy as np
import os

def load_image (mode=0):
    dirs = os.getcwd().split(os.sep)
    dirs.pop()
    path = os.sep.join(dirs)+os.sep+'images/binary_A.png'
    
    print(path)
    img = cv.imread(path,mode)
    if img is None:
        print("failed to load image")
        exit(1)
    return img

img = load_image()
kernel= np.ones((3,3),np.uint8)

erosion = cv.erode(img,kernel,iterations=4)
dilation = cv.dilate(img,kernel,iterations=4)

opening = cv.morphologyEx(img,cv.MORPH_OPEN,kernel)

close = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)
plt.figure(figsize=(12,8))

plt.subplot(2,3,1);plt.title("Errosion");plt.imshow(erosion,cmap='gray')
plt.subplot(2,3,2);plt.title("Dillation");plt.imshow(dilation,cmap='gray')

plt.subplot(2,3,3);plt.title("Opening");plt.imshow(opening,cmap='gray')
plt.subplot(2,3,4);plt.title("Close");plt.imshow(close,cmap='gray')
plt.show()
