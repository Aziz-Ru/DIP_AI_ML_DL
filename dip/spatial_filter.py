import numpy as np

def custom_filter2D(image, kernel, mode='same'):
    """
    Applies a 2D convolution to an image, similar to cv2.filter2D.

    Args:
        image (np.ndarray): The input image (grayscale).
        kernel (np.ndarray): The convolution kernel.
        mode (str): 'same' or 'valid'.
                    'same': output size is the same as input size, padding with zeros.
                    'valid': output size is smaller, only computes where kernel fully overlaps.

    Returns:
        np.ndarray: The filtered image.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if mode == 'same':
        # Calculate padding to maintain same output size
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output_height, output_width = image_height, image_width
    elif mode == 'valid':
        padded_image = image
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
    else:
        raise ValueError("Mode must be 'same' or 'valid'.")

    output_image = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and sum
            output_image[i, j] = np.sum(roi * kernel)

    return output_image


