
import  matplotlib.pyplot as  plt
import cv2 as cv
import numpy as np
import os


img = np.zeros((256,256,3))

plt.figure(figsize=(12,8))

plt.subplot(3,3,1)
plt.imshow(img)
plt.axis('off')

img[:]=(255,0,0)

plt.subplot(3,3,2)
plt.imshow(img)
plt.axis('off')



img[:]=(0,255,0)

plt.subplot(3,3,3)
plt.imshow(img)
plt.axis('off')

img[:]=(0,0,255)

plt.subplot(3,3,4)
plt.imshow(img)
plt.axis('off')


img[:]=(255,255,0)

plt.subplot(3,3,5)
plt.imshow(img)
plt.axis('off')

img[:]=(255,0,255)

plt.subplot(3,3,6)
plt.imshow(img)
plt.axis('off')



img[:]=(0,255,255)
plt.subplot(3,3,7)
plt.imshow(img)
plt.axis('off')

plt.show()
