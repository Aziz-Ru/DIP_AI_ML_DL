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
    plt.subplot(5,3,1)
    plt.title("RGB IMAGE")
    plt.imshow(rgb)
    plt.subplot(5,3,2)
    plt.title("Red channel",)
    plt.imshow(r,cmap='Reds')
    plt.subplot(5,3,3)
    plt.title("Green Channel")
    plt.imshow(g,cmap='Greens')
    plt.subplot(5,3,4)
    plt.title("Blue channel")
    plt.imshow(b,cmap='Blues')

    rw=r*0.299
    gw=g*0.587
    bw=b*0.114

    plt.subplot(5,3,5)
    plt.title("Red * 0.299")
    plt.imshow(rw,)

    plt.subplot(5,3,6)
    plt.title("Green * 0.144")
    plt.imshow(gw,)

    plt.subplot(5,3,7)
    plt.title("Blue * 0.587")
    plt.imshow(bw,)

    final_img = 0.299*r + 0.587*g + 0.114*b
    plt.subplot(5,3,8)
    plt.title("Gray Image")
    plt.imshow(final_img,cmap='gray')

    plt.figure(figsize=(15,10))
    plt.subplot(3,1,1)
    plt.title("Red Histogram")
    r_histogram=np.zeros(256,dtype=int)
    for val in r.flatten():
        r_histogram[val]+=1
    plt.plot(range(256), r_histogram,color='red')

    plt.subplot(3,1,2)
    plt.title("Blue Histogram")
    g_histogram=np.zeros(256,dtype=int)
    for val in g.flatten():
        g_histogram[val]+=1
    plt.plot(range(256), g_histogram,color='blue')

    plt.subplot(3,1,3)
    plt.title("Green Histogram")
    b_histogram=np.zeros(256,dtype=int)
    for val in b.flatten():
        b_histogram[val]+=1
    plt.plot(range(256), b_histogram,color='green')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    build_histrogram()