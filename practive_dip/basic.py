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

gray_img = load_image()
bgr_img = load_image(mode=1)
rgb_img = cv.cvtColor(bgr_img,cv.COLOR_BGR2RGB)

hist = cv.calcHist([gray_img],[0],None,[256],[0,256])

resize_img = cv.resize(rgb_img,(128,128))
canny_img = cv.Canny(gray_img,threshold1=50,threshold2=200)
plt.figure(figsize=(10,5))

plt.subplot(2,3,1)
plt.imshow(gray_img,cmap='gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(rgb_img)
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(resize_img)
plt.title("Resize Image")
plt.axis('off')


plt.subplot(2,3,4)
plt.imshow(canny_img,cmap='gray')
plt.title("Canny Image")
plt.axis('off')

plt.subplot(2,3,5)
plt.hist(gray_img.ravel(),bins=256,range=[0,256])
plt.xlim([0,256])
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2,3,6)
plt.plot(hist)
plt.xlim([0,256])


plt.show()

