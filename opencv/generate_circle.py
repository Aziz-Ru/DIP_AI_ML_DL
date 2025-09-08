
import numpy as np
import cv2 as cv
import os
img=np.zeros((200,500,3),dtype='uint8')

circle=cv.circle(img.copy(),(250,100),80,(255,255,255), thickness=-1)

cv.imwrite(os.path.join(os.getcwd(),'image/circle.jpg'),circle)
