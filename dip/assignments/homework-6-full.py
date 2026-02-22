import numpy as np
import cv2 as cv
from spatial_filter import custom_filter2D
import matplotlib.pyplot as plt

from kernals import (
    smoothing_kernel,
    sobel_x, sobel_y,
    prewitt_x, prewitt_y,
    scharr_x, scharr_y,
    laplace_kernel,
    custom_kernel_1, custom_kernel_2, custom_kernel_3, custom_kernel_4
)



if __name__ == "__main__":
    
    test_image = cv.imread('image/tulip.jpeg', cv.IMREAD_GRAYSCALE).astype(np.float32)
    kernels = {
        "smoothing": smoothing_kernel,
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "prewitt_x": prewitt_x,
        "prewitt_y": prewitt_y,
        "scharr_x": scharr_x,
        "scharr_y": scharr_y,
        "laplace": laplace_kernel,
        "custom_1": custom_kernel_1,
        "custom_2": custom_kernel_2,
        "custom_3": custom_kernel_3,
        "custom_4": custom_kernel_4,
    }

    plt.figure(figsize=(15, 15))
    index = 1
    
    for name,kernel in kernels.items():
        filtered_image = custom_filter2D(test_image, kernel, mode='same')
        filtered_image_valid = custom_filter2D(test_image, kernel, mode='valid')
        plt.subplot(5, 5, index)
        plt.title(f'Filter: {name} (same)')
        plt.imshow(filtered_image, cmap='gray')
        plt.axis('off')
        index += 1
        plt.subplot(5, 5, index)
        plt.title(f'Filter: {name} (valid)')
        plt.imshow(filtered_image_valid, cmap='gray')
        plt.axis('off')
        index += 1


    plt.tight_layout()
    plt.show()

