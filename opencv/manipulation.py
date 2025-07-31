import cv2 as cv 
import matplotlib.pyplot as plt 
import os

def show_image():
  imgpath1 = os.path.join(os.getcwd(), 'image/checkerboard_18.png')
  img = cv.imread(imgpath1, 1)
  b,g,r=cv.split(img)
  
  img_copy=img.copy()
  img_copy[2,2]=200
  img_copy[2,3]=200
  img_copy[3,2]=200
  img_copy[3,3]=200
  plt.figure(figsize=(10,10))

  plt.subplot(3,2,1)
  plt.title("BGR IMAGE")
  plt.imshow(img_copy,cmap='gray')

  plt.show()

if __name__=='__main__':
  show_image()