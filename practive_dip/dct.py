
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


img = load_image()

img = np.float32(img)/255.0
dct = cv.dct(img)
dct_log = np.log(np.abs(img)+1)

img_b= cv.idct(dct)

plt.figure(figsize=(12,8))
plt.subplot(1,3,1);plt.imshow(img);plt.title("Original")
plt.subplot(1,3,2);plt.imshow(dct_log);plt.title("DCT Spectrum")

plt.subplot(1,3,3);plt.imshow(img_b);plt.title("Reconstructed")


plt.show()

