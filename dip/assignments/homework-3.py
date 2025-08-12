import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Load grayscale image
img_path = os.path.join(os.getcwd(), 'image/tulip.jpeg')  # Change path if needed
img = cv.imread(img_path, 0)

plt.figure(figsize=(20, 10))


plt.subplot(3, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

for bit in range(8):
    mask = 1 << bit
    # print(mask)
    bit_plane = cv.bitwise_and(img, mask)
    bit_plane_scaled = bit_plane * 255 // mask
    
    plt.subplot(3, 4, bit + 2)
    plt.imshow(bit_plane_scaled, cmap='gray')
    plt.title(f"Bit Plane {bit}")
    plt.axis("off")

plt.tight_layout()
plt.show()
