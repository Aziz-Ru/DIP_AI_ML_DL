import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

imgpath = os.path.join(os.getcwd(), 'image/tulip.jpeg')
img = cv.imread(imgpath, 0)
rows,cols=img.shape

def step_func(threshold):
  new_img=np.zeros((rows,cols),dtype=np.uint8)
  for i in range(rows):
    for j in range(cols):
      value=img[i,j]
      if(value>threshold):
        new_img[i,j]=255
  return new_img

new_img1=step_func(80)
new_img2=step_func(120)
new_img3=step_func(180)

plt.figure(figsize=(20,20))
plt.subplot(5,3,1)
plt.imshow(new_img1,cmap='gray')
plt.axis('off')
plt.title("Step function Threshold value 80")

plt.subplot(5,3,2)
plt.imshow(new_img2,cmap='gray')
plt.axis('off')
plt.title("Step function Threshold value 120")

plt.subplot(5,3,3)
plt.imshow(new_img3,cmap='gray')
plt.axis('off')
plt.title("Step function Threshold value 180")

hits1=cv.calcHist([new_img1],[0],None,[256],[0,256])
hits2=cv.calcHist([new_img2],[0],None,[256],[0,256])
hits3=cv.calcHist([new_img3],[0],None,[256],[0,256])
plt.subplot(5,3,4)
plt.plot(hits1)
plt.axis('off')
plt.title("Threshold value 80 Histogram")


plt.subplot(5,3,5)
plt.plot(hits2)
plt.axis('off')
plt.title("Threshold value 120 Histogram")

plt.subplot(5,3,6)
plt.plot(hits3)
plt.axis('off')
plt.title("Threshold value 180 Histogram")


def f1(x):
  if(x<100):
    return x
  elif 100<=x<220:
    return 180
  else:
    return 140

def transformf1():
  new_img=np.zeros((rows,cols),dtype=np.uint8)
  for i in range(rows):
    for j in range(cols):
      value=img[i,j]
      new_img[i,j]=f1(value)
  return new_img

   
x1_val=np.linspace(0,255,255)
y1_val=[f1(x) for x in x1_val]
trans_img1=transformf1()
trans_hist1=cv.calcHist([trans_img1],[0],None,[256],[0,256])

plt.subplot(5,3,7)
plt.plot(x1_val,y1_val)
plt.axis('off')
plt.title("Transform function -1")
plt.axhline(0)
plt.axvline(0)


plt.subplot(5,3,8)
plt.imshow(trans_img1,cmap='gray')
plt.axis('off')
plt.title("Transform function-1 Image")

plt.subplot(5,3,9)
plt.plot(trans_hist1)
plt.axis('off')
plt.title("Transform function-1 Historgram")

def f2(x):
  if(x<50):
    return 20
  elif 50<=x<90:
    return 200
  elif 90<=x<150:
    return x
  elif 150<=x<210:
    return 180
  elif 210<=x<256:
    return 250
  
def transformf2():
  new_img=np.zeros((rows,cols),dtype=np.uint8)
  for i in range(rows):
    for j in range(cols):
      value=img[i,j]
      new_img[i,j]=f2(value)
  return new_img

x2_val=np.linspace(0,255,255)
y2_val=[f2(x) for x in x1_val]
trans_img2=transformf2()
trans_hist2=cv.calcHist([trans_img2],[0],None,[256],[0,256])

plt.subplot(5,3,10)
plt.plot(x2_val,y2_val)
plt.axis('off')
plt.title("Transform function -2")
plt.axhline(0)
plt.axvline(0)


plt.subplot(5,3,11)
plt.imshow(trans_img2,cmap='gray')
plt.axis('off')
plt.title("Transform function-2 Image")

plt.subplot(5,3,12)
plt.plot(trans_hist2)
plt.axis('off')
plt.title("Transform function-2 Historgram")

plt.show()