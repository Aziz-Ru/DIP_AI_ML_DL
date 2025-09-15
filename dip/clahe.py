import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def build_histrogram():
    print(os.getcwd())
    imgpath1 = os.path.join(os.getcwd(), 'image/tulip.jpeg')
    imgpath2= os.path.join(os.getcwd(), 'image/paddy_image.jpeg')
    img1 = cv.imread(imgpath1, 0)
    img2 = cv.imread(imgpath2, 0)

    calhe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = calhe.apply(img1)
    cl2 = calhe.apply(img2)

    hist = cv.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv.calcHist([cl1], [0], None, [256], [0,256])

    hist3 = cv.calcHist([img2], [0], None, [256], [0,256])
    hist4 = cv.calcHist([cl2], [0], None, [256], [0,256])

    plt.figure(figsize=(10,10))
    plt.subplot(4,2,1)
    plt.title("Original Image")
    plt.imshow(img1, cmap='gray')
    plt.axis("off")
    plt.subplot(4,2,2)
    plt.title("CLAHE Image")
    plt.imshow(cl1, cmap='gray')
    plt.axis("off")
    plt.subplot(4,2,3)
    plt.title("Original Histogram")
    plt.plot(hist)
    plt.xlim([0,256])
    plt.subplot(4,2,4)
    plt.title("CLAHE Histogram")
    plt.plot(hist2)
    plt.xlim([0,256])
    plt.subplot(4,2,5)
    plt.title("Original Image 2")
    plt.imshow(img2, cmap='gray')
    plt.axis("off")
    plt.subplot(4,2,6)
    plt.title("CLAHE Image 2")
    plt.imshow(cl2, cmap='gray')
    plt.axis("off")

    plt.subplot(4,2,7)
    plt.title("Original Histogram 2")
    plt.plot(hist3)
    plt.xlim([0,256])
    plt.subplot(4,2,8)
    plt.title("CLAHE Histogram 2")
    plt.plot(hist4)
    plt.xlim([0,256])
    


    plt.show()

if __name__ == '__main__':
    build_histrogram()