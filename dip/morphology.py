import cv2
import numpy as np
import matplotlib.pyplot as plt

binary = cv2.imread("image/binary_A.png", 0)

# _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=3)
dilation = cv2.dilate(binary, kernel, iterations=3)


opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)


# Show results
titles = ['Original', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient']
images = [binary, erosion, dilation, opening, closing, gradient]

plt.figure(figsize=(10,6))
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
