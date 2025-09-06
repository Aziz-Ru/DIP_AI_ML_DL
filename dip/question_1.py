import cv2 as cv 
import matplotlib.pyplot as plt 
import os

def solve():
    img1 = cv.imread('image/New_Zealand_Lake.jpg',0)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2=cv.imread('image/sky.jpeg',0)

    img11=cv.resize(img1,(512,512))
    img22=cv.resize(img2,(512,512))
    
    merge_img=img11+img22
    plt.figure(figsize=(10,10))

    plt.subplot(3,2,1)
    plt.imshow(img1,cmap='gray')
    plt.subplot(3,2,2)
    plt.imshow(img22, cmap='gray')
    plt.subplot(3,2,3)
    plt.imshow(img11, cmap='gray')
    plt.subplot(3,2,4)
    plt.imshow(img22, cmap='gray')

    plt.subplot(3,2,5)
    plt.imshow(img22, cmap='gray')
    plt.show()

if __name__=='__main__':
    solve()