import cv2 as cv 
import matplotlib.pyplot as plt 
import os
import numpy as np

def solve():
    img1 = cv.imread('image/New_Zealand_Lake.jpg',0)
    img2=cv.imread('image/sky.jpeg',0)
    mx,my=img1.shape
    lx,ly= img2.shape
    dx=(mx-lx)//2
    dy = (my-ly)//2
    new_img = np.zeros((mx,my),dtype=np.uint8)
    for i in range(mx):
        for j in range(my):
            if(j>dx and j<dx+ly and i>dy and i<dy+lx):
                new_img[i,j] = min(255,img1[i,j]+img2[i-dy,j-dx])
    
    plt.figure(figsize=(10,10))
    plt.subplot(3,2,1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(3,2,2)
    plt.imshow(img2, cmap='gray')
    plt.subplot(3,2,3)
    plt.imshow(new_img, cmap='gray')
    plt.show()

if __name__=='__main__':
    solve()