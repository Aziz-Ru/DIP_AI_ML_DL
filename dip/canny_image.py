import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def build_histrogram():
    print(os.getcwd())
    imgpath1 = os.path.join(os.getcwd(), 'image/tulip.jpeg')
    imgpath2 = os.path.join(os.getcwd(), 'image/paddy_image.jpeg')
    img = cv.imread(imgpath1, 0)
    img2= cv.imread(imgpath2, 0)
    if img is None:
        print("Failed to load image")
        return
    edges1 = cv.Canny(img, 100, 200)
    edges2 = cv.Canny(img2, 50, 150)
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.subplot(2,2,2)
    plt.title("Canny Edges")
    plt.imshow(edges1, cmap='gray')
    plt.axis("off")
    plt.subplot(2,2,3)
    plt.title("Canny Edges 2")
    plt.imshow(edges2, cmap='gray')
    plt.axis("off")
    
    


    plt.show()

if __name__ == '__main__':
    build_histrogram()