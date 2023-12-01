import cv2
import numpy as np

# Read the image
image = cv2.imread('test1dirt.png')  # Replace with your image path
original_image = image.copy()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)

# Apply GaussianBlur to reduce noise and help with edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Blurred', blurred)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow('All Edges', edges)

# Create a region of interest (ROI) mask to focus on the road area
height, width = edges.shape
roi_vertices = np.array([[(0, height), (width, height), (width, height // 2), (0, height // 2)]], dtype=np.int32)
mask = np.zeros_like(edges)
cv2.fillPoly(mask, roi_vertices, 255)

# Apply the ROI mask to the edges
masked_edges = cv2.bitwise_and(edges, mask)
cv2.imshow('Masked Edges', masked_edges)

# Apply Hough Line Transformation to detect lines in the image
lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

# Draw the detected lines on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the results
cv2.imshow('Original Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
