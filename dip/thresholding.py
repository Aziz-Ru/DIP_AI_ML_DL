import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

imgpath1 = os.path.join(os.getcwd(), 'image/paddy_image.jpeg')
img = cv.imread(imgpath1, 0)
row,col=img.shape

def step_func(value):
  thershold_img=np.zeros((row,col))
  for  i in range(row):
    for j in range(col):
      if img[i,j]<value:
        thershold_img[i,j]=0
      else:
        thershold_img[i,j]=255

  return thershold_img


def custom_fun(lower,higher):
  thershold_img=np.zeros((row,col))
  for  i in range(row):
    for j in range(col):
      if img[i,j]<lower:
        thershold_img[i,j]=img[i,j]
      elif img[i,j]>higher:
        thershold_img[i,j]=225
      else:
        thershold_img[i,j]=150

  return thershold_img

plt.figure(figsize=(10,10))
plt.subplot(3,3,1)
plt.title("Step function thereshold:80")
plt.imshow(step_func(value=80),cmap='grey')
plt.axis('off')
plt.subplot(3,3,2)
plt.title("Step function thereshold:120")
plt.imshow(step_func(value=120),cmap='grey')
plt.axis('off')
plt.subplot(3,3,3)
plt.title("Step function thereshold:180")
plt.imshow(step_func(value=180),cmap='grey')
plt.axis('off')

plt.subplot(3,3,4)
plt.title("Custom function thereshold:60,200")
plt.imshow(custom_fun(lower=60,higher=200),cmap='grey')
plt.axis('off')

plt.subplot(3,3,5)
plt.title("Custom function thereshold:120,150")
plt.imshow(custom_fun(lower=120,higher=150),cmap='grey')
plt.axis('off')

plt.subplot(3,3,6)
plt.title("Custom function thereshold:60,180")
plt.imshow(custom_fun(lower=90,higher=180),cmap='grey')
plt.axis('off')

x = img.astype(np.float32) / 255.0

y = 2 * (x ** 2) - 4 * x + 1

thresh_img = np.where(y >= 0.2, 255, 0).astype(np.uint8)
plt.subplot(3,3,7)
plt.title("Custom function thereshold")
plt.imshow(thresh_img,cmap='grey')
plt.axis('off')

thresh_img = np.where(y >= 0.3, 255, 0).astype(np.uint8)
plt.subplot(3,3,8)
plt.title("Custom function thereshold")
plt.imshow(thresh_img,cmap='grey')
plt.axis('off')

thresh_img = np.where(y >= 0.9, 255, 0).astype(np.uint8)
plt.subplot(3,3,9)
plt.title("Custom function thereshold")
plt.imshow(thresh_img,cmap='grey')
plt.axis('off')

plt.show()

