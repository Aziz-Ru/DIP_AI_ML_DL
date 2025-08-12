import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

rectangle = np.zeros((200, 500, 3), dtype='uint8')

# plt.figure(figsize=(10, 5))

# # Plot at i = 0, 50, 100, 150, 200, 250
# for idx, i in enumerate(range(0, 256, 50), start=1):
#     change_img = cv.add(rectangle, i)  # returns a new image
#     plt.subplot(2, 3, idx)
#     plt.imshow(change_img)
#     plt.title(f"Brightness +{i}")
#     plt.axis('off')

tmp_img = np.zeros((256, 256, 3), dtype=np.uint8)


for i in range(255):
    for j in range(255):
        tmp_img[i,j,0] = j
        tmp_img[i,j,1] = 0
        tmp_img[i,j,2] = 0

plt.figure(figsize=(10,10))
plt.subplot(3,3,1)

plt.imshow(tmp_img)

for i in range(255):
    for j in range(255):
        tmp_img[i,j,0] = 0
        tmp_img[i,j,1] = 0
        tmp_img[i,j,2] = j



rectangle[:,:,0]=255
plt.subplot(3,3,2)
plt.title("Red")
plt.imshow(rectangle)

rectangle[:,:,0]=0
rectangle[:,:,1]=255
plt.subplot(3,3,3)
plt.title("Green")
plt.imshow(rectangle)

rectangle[:,:,0]=0
rectangle[:,:,1]=0
rectangle[:,:,2]=255
plt.subplot(3,3,4)
plt.title("Blue")
plt.imshow(rectangle)

rectangle[:,:,0]=0
rectangle[:,:,1]=255
rectangle[:,:,2]=255
plt.subplot(3,3,5)
plt.title("Green + Blue")
plt.imshow(rectangle)

rectangle[:,:,0]=255
rectangle[:,:,1]=0
rectangle[:,:,2]=255
plt.subplot(3,3,6)
plt.title("Red+Blue")
plt.imshow(rectangle)

plt.subplot(3,3,7)
plt.imshow(tmp_img)

for i in range(255):
    for j in range(255):
        tmp_img[i,j,0] = 0
        tmp_img[i,j,1] = j
        tmp_img[i,j,2] = 0

plt.subplot(3,3,8)
plt.imshow(tmp_img)

check_img=np.zeros((256,256,3),dtype=np.uint8)

for channel in range(3):
    for i in range(256):
        for j in range(256):
            check_img[i,j,channel]=100
            check_img[i,j,channel]=200
            check_img[i,j,channel]=10
plt.imshow(check_img)
plt.tight_layout()
plt.show()
