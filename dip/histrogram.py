import cv2 as cv 
import matplotlib.pyplot as plt 
import os
import numpy as np

def build_histrogram():
    print(os.getcwd())
    imgpath1 = os.path.join(os.getcwd(), 'image/flower.jpeg')
    img = cv.imread(imgpath1, 1)
    if img is None:
        print("Failed to load image")
        return
    
    rgb=img[:,:,::-1]
    r,g,b=cv.split(rgb)
    # plt.figure(figsize=(15, 5))
    colors = ['r', 'g', 'b']
    
    # for channel in range(rgbimg.shape[2]):
    #     cnt = np.zeros(256, dtype=int)
    #     for row in rgbimg[:, :, channel]:
    #         for pixel in row:
    #             cnt[pixel] += 1
    #     x = np.arange(256)
    #     plt.subplot(1, 3, channel + 1)
    #     plt.stem(x, cnt, linefmt=colors[channel], markerfmt=colors[channel]+'o', basefmt=" ")
    #     plt.title(f'Channel {channel} ({colors[channel].upper()})')
    #     plt.xlabel('Pixel Value')
    #     plt.ylabel('Count')
    #     plt.xlim(0, 255)
    plt.figure(figsize=(15,10))
    plt.subplot(3,2,1)
    plt.title("RGB IMAGE")
    plt.imshow(rgb)
    plt.subplot(3,2,2)
    plt.title("Red channel")
    plt.imshow(r)
    plt.subplot(3,2,3)
    plt.title("Green Channel")
    plt.imshow(g)
    plt.subplot(3,2,4)
    plt.title("Blue channel")
    plt.imshow(b)
    r=r*0.299
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    build_histrogram()