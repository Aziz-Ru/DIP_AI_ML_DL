
import  matplotlib.pyplot as  plt
import cv2 as cv
import numpy as np
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

def he(img):
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    pdf= hist/hist.sum()
    cdf=pdf.cumsum()

    look=(cdf*255).astype(np.uint8)
    he_img= look[img]
    return he_img

img = load_image()

equlized = cv.equalizeHist(img)

eq=he(img)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original IMG")


plt.subplot(2,2,2)
plt.imshow(equlized)
plt.title("Equalized IMG")

plt.subplot(2,2,3)
plt.imshow(eq)
plt.title("Eq IMG")

plt.show()
