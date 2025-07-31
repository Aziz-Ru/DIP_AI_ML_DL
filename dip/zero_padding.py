import cv2 as cv 
import matplotlib.pyplot as plt 
import os
import numpy as np

def build_zeroPad():
    print(os.getcwd())
    imgpath1 = os.path.join(os.getcwd(), 'image/image-4.jpeg')
    imgpath2=  os.path.join(os.getcwd(), 'checkerboard_color.png')
    img1 = cv.imread(imgpath1, 0)
    big_img= cv.imread(imgpath2,0)
    if img1 is None or big_img is None:
        print("Failed to load image")
        return
    
    height,width=big_img.shape

    def image_padd(img,tr,tc):
        no_of_rows,no_of_cols=img.shape
        
        row_diff=(tr-no_of_rows)//2
        col_diff=(tc-no_of_cols)//2
        print(no_of_cols)
        pad_img=np.zeros((tr,tc),dtype=np.uint8)
        for row in range (no_of_rows):
            for col in range (no_of_cols):
                if row>row_diff and col>col_diff and row<no_of_rows and col<no_of_cols:
                    pad_img[row][col]=img[row-row_diff][col-col_diff]
                else:
                     pad_img[row][col]=0
        
        return pad_img

    pad_img1=image_padd(img1,height,width)
    pad_img2=image_padd(big_img,height,width)
    plt.figure(figsize=(10,10))
    plt.subplot(3,2,1)
    plt.imshow(pad_img1)

    plt.subplot(3,2,2)
    plt.imshow(pad_img2)

    plt.subplot(3,2,3)
    plt.imshow(pad_img1+pad_img2)

    plt.subplot(3,2,4)
    plt.imshow(img1)

    plt.show()

