
import numpy as np
import cv2 as cv
import os
img=np.zeros((296,474,3),dtype='uint8')

circle=cv.circle(img.copy(),(237,148),100,(255,255,255), thickness=-1)

cv.imwrite(os.path.join(os.getcwd(),'image/circle_02.jpg'),circle)
