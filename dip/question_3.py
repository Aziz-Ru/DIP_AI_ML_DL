import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# Load image in grayscale
imgpath = os.path.join(os.getcwd(), 'image/einstein.jpeg')
img = cv.imread(imgpath, 0)

if img is None:
    raise FileNotFoundError("Image not found. Check the path.")

hist = cv.calcHist([img], [0], None, [256], [0, 256])

# Plot histogram
plt.figure(figsize=(8, 4))
plt.subplot(2,1,1)
plt.imshow(img,cmap='gray')
plt.subplot(2,1,2)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(hist, color='black')
plt.xlim([0, 256])
# plt.axis('off')
plt.show()
