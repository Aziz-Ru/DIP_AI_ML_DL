import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Load grayscale image
img_path = os.path.join(os.getcwd(), 'image/tulip.jpeg')  # Change path if needed
img = cv.imread(img_path, 0)

plt.figure(figsize=(20, 10))

plt.subplot(3,3,1)
plt.imshow(img.astype(np.uint8), cmap='gray')
plt.title('Original Image')
plt.axis('off')

avg_kernal=np.ones((3,3),np.float32)/9
smoothed = cv.filter2D(img,-1,avg_kernal)

sobelx_kernal = [[-1,0,1],[-2,0,2],[-1,0,1]]
sobely_kernal = [[-1,-2,-1],[0,0,0],[1,2,1]]

sobel_x = cv.filter2D(smoothed, -1, np.array(sobelx_kernal))
sobel_y = cv.filter2D(smoothed, -1, np.array(sobely_kernal))
plt.subplot(3,3,2)
plt.imshow(sobel_x.astype(np.uint8), cmap='gray')
plt.title('Sobel X after Smoothing')
plt.axis('off')
plt.subplot(3,3,3)
plt.imshow(sobel_y.astype(np.uint8), cmap='gray')
plt.title('Sobel Y after Smoothing')
plt.axis('off')


Prewitt_x_kernal = [[-1,0,1],[-1,0,1],[-1,0,1]]
Prewitt_y_kernal = [[-1,-1,-1],[0,0,0],[1,1,1]]
prewitt_x = cv.filter2D(img, -1, np.array(Prewitt_x_kernal))
prewitt_y = cv.filter2D(img, -1, np.array(Prewitt_y_kernal))

plt.subplot(3,3,4)
plt.imshow(prewitt_x.astype(np.uint8), cmap='gray')
plt.title('Prewitt X')
plt.axis('off')

plt.subplot(3,3,5)
plt.imshow(prewitt_y.astype(np.uint8), cmap='gray')
plt.title('Prewitt Y')
plt.axis('off')

lapacian_kernal = [[0,-1,0],[-1,4,-1],[0,-1,0]]
laplacian = cv.filter2D(img, -1, np.array(lapacian_kernal))

plt.subplot(3,3,6)
plt.imshow(laplacian.astype(np.uint8), cmap='gray')
plt.title('Laplacian')
plt.axis('off')

# plt.subplot(1,3,3)
# plt.imshow(prewitt_y.astype(np.uint8), cmap='gray')
# plt.title('Prewitt Y')
# plt.axis('off')

plt.show()