import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv


def show_image():
  imgpath1 = os.path.join(os.getcwd(), 'image/flower.jpeg')
  bgrIMG = cv.imread(imgpath1, 1)
  # img = cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
  rgbImg=bgrIMG[:,:,::-1]
  # rgbImg.copy() returns a new, independent copy 
  line_img=cv.line(rgbImg.copy(),(100,100),(200,100),(0,200,200),thickness=15)
  rectangle_img = cv.rectangle(rgbImg.copy(),(10,20),(100,100),(255,0,0),thickness=-4)
    #img = cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])

    #  img: The output image that has been annotated.

    #The function has 4 required arguments:

    #img: Image on which we will draw a line

    #center: Center of the circle

    #radius: Radius of the circle

   # color: Color of the circle which will be drawn

    # Next, let's check out the (optional) arguments which we are going to use quite extensi
    #ely. thickness: Thickness of the circle outline (if positive). If a negative value is s
    #pplied for this argument, it will result in a filled circle.
    #lineType: Type of the circle boundary. This is exact same as lineType argument in cv2.line

  circle_img=cv.circle(rgbImg.copy(),(100,100),50,(0,200,200),thickness=-5)
  plt.figure(figsize=(10,10))

  plt.subplot(2,2,1)
  plt.title("Draw Line")
  plt.imshow(line_img)
  
  # img = cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
  # img: Image on which we will draw a line
  # pt1: First point(x,y location) of the line segment
  # pt2: Second point of the line segment
  #  color: Color of the line which will be drawn

  plt.subplot(2,2,2)
  plt.title("Draw Circle")
  plt.imshow(circle_img)
  
  plt.subplot(2,2,3)
  plt.title("Rectangle")
  plt.imshow(rectangle_img)

  plt.show()


if __name__=='__main__':
    show_image()
