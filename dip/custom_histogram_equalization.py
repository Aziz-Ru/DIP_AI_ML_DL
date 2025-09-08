import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_histogram_equalization(img):
    """
    Perform histogram equalization from scratch.
    Args:
        img: Grayscale image (numpy array)
    Returns:
        equalized image
    """
    # Flatten image to 1D
    flat = img.flatten()
    
    # Step 1: Histogram
    hist = np.bincount(flat, minlength=256)
    
    # Step 2: PDF (probability density function)
    pdf = hist / np.sum(hist)
    
    # Step 3: CDF (cumulative distribution function)
    cdf = np.cumsum(pdf)
    
    # Step 4: Transform
    equalized = np.floor(255 * cdf[flat]).astype(np.uint8)
    
    # Reshape to original image
    return equalized.reshape(img.shape)


# -----------------------
# Load image (grayscale)
# -----------------------
img = cv2.imread("image/tulip.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply custom function
custom_eq = custom_histogram_equalization(img)

# Apply OpenCV function
cv_eq = cv2.equalizeHist(img)

# -----------------------
# Display results
# -----------------------
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")

plt.subplot(2,3,2)
plt.title("Custom Equalized")
plt.imshow(custom_eq, cmap="gray")

plt.subplot(2,3,3)
plt.title("OpenCV Equalized")
plt.imshow(cv_eq, cmap="gray")

# Histograms
plt.subplot(2,3,4)
plt.title("Original Histogram")
plt.hist(img.ravel(), bins=256, range=(0,256))

plt.subplot(2,3,5)
plt.title("Custom Equalized Histogram")
plt.hist(custom_eq.ravel(), bins=256, range=(0,256))

plt.subplot(2,3,6)
plt.title("OpenCV Equalized Histogram")
plt.hist(cv_eq.ravel(), bins=256, range=(0,256))

plt.tight_layout()
plt.show()
