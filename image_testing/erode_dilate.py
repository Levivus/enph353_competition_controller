import cv2
import numpy as np

# Read the image
image = cv2.imread('hsv_threshold.png', cv2.IMREAD_GRAYSCALE)

# Define a kernel (structuring element)
kernel = np.ones((3, 3), np.uint8)

# Perform erosion
eroded_image = cv2.erode(image, kernel, iterations=6)

# Perform dilation
dilated_image = cv2.dilate(eroded_image, kernel, iterations=20)

# Display the original, dilated, and eroded images
cv2.imshow('Original Image', image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
