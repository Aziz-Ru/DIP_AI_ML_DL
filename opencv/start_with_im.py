import cv2 as cv 
import matplotlib.pyplot as plt 
import os

def show_image():
  imgpath1 = os.path.join(os.getcwd(), 'image/coca-cola-logo.png')
  img = cv.imread(imgpath1, 1)
  # 
  b,g,r=cv.split(img)
  rgbImg = img[:,:,::-1]
  built_inrgb=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  
  plt.figure(figsize=(10,10))

  plt.subplot(3,2,1)
  plt.title("BGR IMAGE")
  plt.imshow(img,cmap='gray')

  plt.subplot(3,2,2)
  plt.imshow(rgbImg,cmap='gray')
  plt.title("RGB IMAGE")

  plt.subplot(3,2,3)
  plt.title("Red Channel")
  plt.imshow(r,cmap='gray')

  plt.subplot(3,2,4)
  plt.title("Green Channel")
  plt.imshow(g,cmap='gray')
  
  plt.subplot(3,2,5)
  plt.title("Blue Channel")
  plt.imshow(b,cmap='gray')

  plt.subplot(3,2,6)
  plt.title("Built in RGB")
  plt.imshow(built_inrgb,cmap='gray')
  plt.show()



if __name__=='__main__':
    show_image()
