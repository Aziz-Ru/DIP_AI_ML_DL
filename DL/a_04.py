import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def part_2():
    cwd = os.getcwd()
    img_path = os.path.join(cwd, "dataset", "person","men", "img0001.jpg")
    img = cv2.imread(img_path,0)
    # img = cv2.cvtColor(img, cv2.)

    # Kernels
    edge = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    blur = np.ones((3,3))/9
    sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

    # Apply filters
    edge_img = cv2.filter2D(img, -1, edge)
    blur_img = cv2.filter2D(img, -1, blur)
    sharp_img = cv2.filter2D(img, -1, sharpen)

    # Show results
    titles = ['Original', 'Edge', 'Blur', 'Sharpen']
    images = [img, edge_img, blur_img, sharp_img]

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.show()

def part_4():

  model = models.vgg16(pretrained=True)

  # First conv layer
  kernels = model.features[0].weight.data

  print(kernels.shape)  # [64, 3, 3, 3]

  # Visualize first 8 kernels
  for i in range(8):
      k = kernels[i].permute(1,2,0)
      plt.subplot(2,4,i+1)
      plt.imshow((k - k.min())/(k.max()-k.min()))
      plt.axis('off')

  plt.show()


if __name__ == "__main__":
    part_2()
    part_4()