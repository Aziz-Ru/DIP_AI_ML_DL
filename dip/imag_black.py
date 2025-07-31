import cv2 as cv 
import matplotlib.pyplot as plt 
import os

def make_black():
  print(os.getcwd())
  imgpath1=os.path.join(os.getcwd(),'coca-cola-logo.png',)
  if imgpath1 is None:
    print("Failed load image")
    return
  img=cv.imread(imgpath1,1)
  rgbimg=cv.cvtColor(img,cv.COLOR_BGR2RGB)
  reverse_img=img[0:,0:,::-1]
  rgbimg[100:300,100:300]=0
    
  plt.figure(figsize=(10,10))
  plt.subplot(3,2,1)
  plt.imshow(rgbimg)

  plt.subplot(3,2,2)
  plt.imshow(reverse_img)

  plt.show()

# Function Syntax

# retval = cv2.imread( filename[, flags] )

# retval: Is the image if it is successfully loaded. Otherwise it is None. This may happen if the filename is wrong or the file is corrupt.

# The function has 1 required input argument and one optional flag:

#    filename: This can be an absolute or relative path. This is a mandatory argument.

#    Flags: These flags are used to read an image in a particular format (for example, grayscale/color/with alpha channel). This is an optional argument with a default value of cv2.IMREAD_COLOR or 1 which loads the image as a color image.

#Before we proceed with some examples, let's also have a look at some of the flags available.

# Flags

#     cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
#     cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
#     cv2.IMREAD_UNCHANGED or -1: Loads image as such including alpha channel.
