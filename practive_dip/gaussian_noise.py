
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

mean=0,
sigma=20

gauss = np.random.normal(mean,sigma,img.shape).astype(np.float32)

noisy_img = img.astype(np.float32)+gauss

noisy_img = np.clip(noisy_img,0,255).astype(np.uint8)


plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img)


plt.subplot(1,2,2)
plt.title("Noisy Image")
plt.imshow(noisy_img)

plt.show()
