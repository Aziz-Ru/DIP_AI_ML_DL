import cv2 as cv 
import matplotlib.pyplot as plt 
import os

def solve():
    img1 = cv.imread('image/New_Zealand_Lake.jpg',0)
    
    black_img= img1.copy()
    for h in range(500):
        for r in range(500):
            black_img[h,r]=0
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(img1, cmap='gray')
    
    plt.subplot(2,2,2)
    plt.imshow(black_img, cmap='gray')
    plt.show()

if __name__=='__main__':
    solve()