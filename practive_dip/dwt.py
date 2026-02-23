

import  matplotlib.pyplot as  plt
import cv2 as cv
import numpy as np
import os
import pywt

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
coeff=pywt.dwt2(img,'haar')

LL,(LH,HL,HH)=coeff

plt.figure(figsize=(12,8))
plt.subplot(2,2,1);plt.title("LL");plt.imshow(LL);
plt.subplot(2,2,2);plt.title("LH");plt.imshow(LH);
plt.subplot(2,2,3);plt.title("HL");plt.imshow(HL);
plt.subplot(2,2,4);plt.title("HH");plt.imshow(HH);




plt.show()
