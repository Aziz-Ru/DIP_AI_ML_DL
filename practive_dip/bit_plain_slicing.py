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

plt.figure(figsize=(12,10))
cnt=1
for i in range(1,8):
    bit_plane = (img>>i) & 1
    plt.subplot(3,3,cnt)
    plt.imshow(bit_plane,cmap='gray')
    plt.title(f"{i}th slicine")
    cnt+=1




plt.show()
