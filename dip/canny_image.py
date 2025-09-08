import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def build_histrogram():
    print(os.getcwd())
    imgpath1 = os.path.join(os.getcwd(), 'image/tulip.jpeg')
    img = cv.imread(imgpath1, 0)
    if img is None:
        print("Failed to load image")
        return
    edges = cv.Canny(img, 100, 200)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Canny Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    build_histrogram()