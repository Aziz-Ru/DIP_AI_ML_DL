import numpy as np

# Smoothing/Average Kernel
smoothing_kernel = np.ones((3, 3)) / 9.0

# Sobel Kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Prewitt Kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])

# Scharr Kernels
scharr_x = np.array([[-3, 0, 3],
                     [-10, 0, 10],
                     [-3, 0, 3]])

scharr_y = np.array([[-3, -10, -3],
                     [0, 0, 0],
                     [3, 10, 3]])

# Laplace Kernel
laplace_kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])

# Custom Kernels
custom_kernel_1 = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]]) # Edge detection

custom_kernel_2 = np.array([[1, 1, 1],
                            [1, -7, 1],
                            [1, 1, 1]]) # Sharpening

custom_kernel_3 = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]) / 5.0 # Horizontal blur

custom_kernel_4 = np.array([[0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0]]) / 5.0 # Vertical blur


