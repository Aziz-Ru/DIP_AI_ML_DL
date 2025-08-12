
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv


def bitwise_operation():
    img1path=os.path.join(os.getcwd(),'image/rectangle.jpg')

    img2path=os.path.join(os.getcwd(),'image/circle.jpg')
    if img1path is None:
        print("failed to load image")
        return
    img1=cv.imread(img1path)
    img2=cv.imread(img2path)
    and_img=cv.bitwise_and(img1,img2)
    or_img=cv.bitwise_or(img1,img2)
    
    xor_img=cv.bitwise_xor(img1,img2)

    plt.figure(figsize=(15,15))
    plt.subplot(3,2,1)
    plt.imshow(img1)

    plt.subplot(3,2,2)
    plt.imshow(img2)

    plt.subplot(3,2,3)
    plt.imshow(and_img)

    plt.subplot(3,2,4)
    plt.imshow(or_img)


    plt.subplot(3,2,5)
    plt.imshow(xor_img)

    plt.show()
    

if __name__=='__main__':
    bitwise_operation()
