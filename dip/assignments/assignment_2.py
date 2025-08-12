import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=np.zeros((256,256,3),dtype=np.uint8)
plt.figure(figsize=(15,12))
colors=["Red","Green","Blue","Yellow","Cyan","Purple"]
cnt=1
step=255//3
for ind,color_name in enumerate(colors):
  for i in range(0,256,step):
    if color_name=='Red':
      img[:,:,0]=i
    elif color_name=='Green':
      img[:,:,1]=i
    elif color_name=='Blue':
      img[:,:,2]=i
    elif color_name=='Yellow':
      img[:,:,0]=i
      img[:,:,1]=i
    elif color_name=='Cyan':
      img[:,:,1]=i
      img[:,:,2]=i
    elif color_name=='Purple':
      img[:,:,0]=i
      img[:,:,2]=i
    plt.subplot(7,4,cnt)
    plt.imshow(img)
    plt.title(f'{color_name} - {i}')
    plt.axis('off')
    cnt+=1
    if color_name=='Red':
      img[:,:,0]=0
    elif color_name=='Green':
      img[:,:,1]=0
    elif color_name=='Blue':
      img[:,:,2]=0
    elif color_name=='Yellow':
      img[:,:,0]=0
      img[:,:,1]=0
    elif color_name=='Cyan':
      img[:,:,1]=0
      img[:,:,2]=0
    elif color_name=='Purple':
      img[:,:,0]=0
      img[:,:,2]=0

for i in range(256):
  for j in range(256):
   img[j,i,0]=i

plt.subplot(7,4,25)
plt.imshow(img)

img[:,:,0]=0
for i in range(256):
  for j in range(256):
   img[j,i,1]=i

plt.subplot(7,4,26)
plt.imshow(img)

img[:,:,1]=0
for i in range(256):
  for j in range(256):
   img[j,i,2]=i

plt.subplot(7,4,27)
plt.imshow(img)

img[:,:,2]=0
for i in range(256):
  for j in range(256):
   img[j,i,0]=i
   img[j,i,1]=i
   img[j,i,2]=i

plt.subplot(7,4,28)
plt.imshow(img)

plt.show()
