
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

def reconstruct(fimg):
    shift= np.fft.ifftshift(fimg)
    img_back=np.fft.ifft2(shift)
    img_back=np.abs(img_back)
    return img_back

img = load_image()

fft= np.fft.fft2(img)
fshift= np.fft.fftshift(fft)
spectrum = np.log(np.abs(fshift))

r,c = img.shape

ml= np.zeros(img.shape)
cv.circle(ml,(r//2,c//2),30,1,-1)
mh=1-ml

lp = ml*fshift
hp = mh*fshift
img_l=reconstruct(lp)
img_h=reconstruct(hp)

plt.figure(figsize=(15,10))
plt.subplot(3,3,1);plt.title("Original");plt.imshow(img,cmap='gray')

plt.subplot(3,3,2);plt.title("Low spectrum");plt.imshow(spectrum,cmap='gray')
plt.subplot(3,3,3);plt.title("Low filter");plt.imshow(img_l,cmap='gray')
plt.subplot(3,3,4);plt.title("High filter");plt.imshow(img_h,cmap='gray')

plt.show()


