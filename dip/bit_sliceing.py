import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
imgpath1 = os.path.join(os.getcwd(), 'image/paddy_image.jpeg')
img = cv.imread(imgpath1, 1)
bit_planes = []
for i in range(8):
    plane = (img >> i) & 1
    bit_planes.append(plane * 255)  # scale 0/1 to 0/255 for display

# Display
plt.figure(figsize=(10,10))
cnt=0
for i, plane in enumerate(bit_planes):
    cnt+=1
    plt.subplot(3,3,cnt)
    plt.imshow(plane)

plt.show()


