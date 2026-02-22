
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

sovelx= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sovely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
prewittx=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitty=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
avg_kernal = np.ones((3,3),np.float32)/9


hsx= cv.filter2D(img,-1,sovelx)
hsy = cv.filter2D(img,-1,sovely)

psx=cv.filter2D(img,-1,prewittx)
psy=cv.filter2D(img,-1,prewitty)

lap = cv.filter2D(img,-1,laplacian)
avg=cv.filter2D(img,-1,avg_kernal)
gaus= cv.GaussianBlur(img,(3,3),sigmaX=1.0)

median = cv.medianBlur(img,3)

plt.figure(figsize=(10,10))

plt.subplot(3,3,1)
plt.imshow(img,cmap='gray')
plt.title("Original Image")


plt.subplot(3,3,2)
plt.imshow(hsx,cmap='gray')
plt.title("Horizontal Sovel Image")

plt.subplot(3,3,3)
plt.imshow(hsy,cmap='gray')
plt.title("Vertical Sovel Image")



plt.subplot(3,3,4)
plt.imshow(gaus,cmap='gray')
plt.title('Gaussian Blur')


plt.subplot(3,3,5)
plt.imshow(psx,cmap='gray')
plt.title("Horizontal Prewitt Image")

plt.subplot(3,3,6)
plt.imshow(psy,cmap='gray')
plt.title("Vertical prewitt Image")

plt.subplot(3,3,7)
plt.imshow(lap,cmap='gray')
plt.title("Laplacian")

plt.subplot(3,3,8)
plt.imshow(avg,cmap='gray')
plt.title('Avarage')


plt.subplot(3,3,9)
plt.imshow(median,cmap='gray')
plt.title('Median')
plt.show()
