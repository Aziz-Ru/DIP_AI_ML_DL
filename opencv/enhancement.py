import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv


def enhancement():
    imgpath=os.path.join(os.getcwd(),'image/flower.jpeg')
    if imgpath is None:
        print("failed to load image")
        return
    bgr=cv.imread(imgpath)
    rgb=bgr[:,:,::-1]
    # Addition or Brightness
    matrix=np.ones(rgb.shape,dtype="uint8")*50
    lighten_img=cv.add(rgb,matrix)
    darker_img=cv.subtract(rgb,matrix)
    plt.figure(figsize=(15,10))
    plt.subplot(5,2,1);plt.title("RGB image");plt.imshow(rgb);

    plt.subplot(5,2,2);plt.title("RGB image");plt.imshow(lighten_img);

    plt.subplot(5,2,3);plt.title("RGB image");plt.imshow(darker_img);
    
    # Multiplication or Contrast
    matrix1=np.ones(rgb.shape)*0.5
    matrix2=np.ones(rgb.shape)*1.2
    low_contrast = np.uint8(cv.multiply(np.float64(rgb),matrix1))
    high_contrast=np.uint8(cv.multiply(np.float64(rgb),matrix2))
    plt.subplot(5,2,4);plt.title("Low Contrast");plt.imshow(low_contrast);
    # The issue is that after multiplying, the values which are already high, are becoming greater than 255. Thus, the overflow issue. How do we overcome this?  #img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))
    plt.subplot(5,2,5);plt.title("High Contrast");plt.imshow(high_contrast);
    plt.show()











if __name__=='__main__':
    enhancement()
