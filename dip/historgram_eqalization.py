import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def build_histrogram():
    print(os.getcwd())
    imgpath1 = os.path.join(os.getcwd(), 'image/einstein.jpeg')
    img = cv.imread(imgpath1, 0)
    if img is None:
        print("Failed to load image")
        return

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    pdf = hist / hist.sum()
    cdf = pdf.cumsum()

    plt.figure(figsize=(10, 5))
    plt.subplot(3,1,1)
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.plot(hist)

    plt.subplot(3,1,2)
    plt.title("PDF")
    plt.xlim(0,255)
    plt.plot(pdf)

    plt.subplot(3,1,3)
    plt.xlim(0,255)
    plt.title("CDF")
    plt.plot(cdf)

    # Show original image with matplotlib
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    # Equalized image (optional)
    # final_img = cv.equalizeHist(img)
    # plt.subplot(2,2,2)
    # plt.title("Equalized Image")
    # plt.imshow(final_img, cmap='gray')
    # plt.axis("off")

    plt.show()

if __name__ == '__main__':
    build_histrogram()
