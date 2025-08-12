import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
imgpath1 = os.path.join(os.getcwd(), 'image/tulip.jpeg')
img = cv.imread(imgpath1, 0)
vertical = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

horizontal = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])

my_kernal = np.array([[1, -2, 1],
               [-2, 2, 1],
               [-1, 1, 1]])

one_by_nine = np.array([[1/9,1/9 , 1/9],
               [1/9, 1/9, 1/9],
               [1/9, 1/9, 1/9]])

random_op=np.multiply(vertical,horizontal)

filter_img1=cv.filter2D(img,-1,vertical)
filter_img2=cv.filter2D(img,-1,horizontal)
filter_img3=cv.filter2D(img,-1,my_kernal)
filter_img4=cv.filter2D(img,-1,one_by_nine)
filter_img5=cv.filter2D(img,-1,random_op)
plt.figure(figsize=(20,20))
plt.subplot(3,3,1)
plt.imshow(filter_img1,cmap='gray')
plt.title("Vertical")
plt.axis('off')

plt.subplot(3,3,2)
plt.imshow(filter_img2,cmap='gray')
plt.title("Horizontal")
plt.axis('off')

plt.subplot(3,3,3)
plt.imshow(img,cmap='gray')
plt.title("Gray")
plt.axis('off')

plt.subplot(3,3,4)
plt.imshow(filter_img3,cmap='gray')
plt.title("own")
plt.axis('off')

plt.subplot(3,3,5)
plt.imshow(filter_img4,cmap='gray')
plt.title("1/9")
plt.axis('off')

plt.subplot(3,3,6)
plt.imshow(filter_img5,cmap='gray')
plt.title("Random")
plt.axis('off')

plt.show()