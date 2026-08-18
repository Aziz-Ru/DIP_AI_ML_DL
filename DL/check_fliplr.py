import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
cwd = os.getcwd()
print(cwd)
img_path = cwd + '/dataset/person/men/img0001.jpg'

img = cv2.imread(img_path,0)

if img is None:
    print("Error: Image not found.")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
flipped_img = np.fliplr(img)
plt.subplot(1,2,2)
plt.imshow(flipped_img, cmap='gray')
plt.title('Flipped Image')
plt.show()
