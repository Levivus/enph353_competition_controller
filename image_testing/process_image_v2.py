import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2 as cv
"""Image processing script
This is the function used in the controller to process the images in the offroad location. It can
be used to test individual images and show them in a for loop at the bottom, to get a clearer idea
of what it's doing, and potentially how it can be improved.
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(script_dir, 'driving_images', 'image_180.png'))

def crop_to_floor(image):
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((2,2),np.uint8)

    # generate floor mask
    lower_hsv = np.array([6,23,118])
    upper_hsv = np.array([84,255,255])

    floor_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # clean up the mask
    floor_mask = cv2.erode(floor_mask,kernel,iterations = 1)
    # generate floor mask
    lower_green = np.array([60, 136, 25])
    upper_green = np.array([75, 145, 35])

    green_mask = cv2.inRange(image, lower_green, upper_green)

    # clean up the mask
    kernel = np.ones((5,5),np.uint8)
    green_mask = cv2.dilate(green_mask,kernel,iterations = 4)

    # set all images outside the mask to black
    image[floor_mask == 0] = 0
    image[green_mask > 0] = 0

    # generate mask for the ROI
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 255

    # find the minimum y value and crop the image to that
    nonzero_points = np.column_stack(np.where(floor_mask > 0))
    min_y = np.min(nonzero_points[:, 0])
    cropped_img = image[min_y:, :]
    cropped_mask = mask[min_y:, :]
    cv2.imshow('Cropped Image', cropped_img)

    return cropped_img, cropped_mask

image, mask = crop_to_floor(image)

# find canny edges
blurred = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
edges = cv2.Canny(blurred, 50, 150)
edges_mask = cv2.Canny(mask, 50, 150)
# dilate the edges mask
kernel = np.ones((2, 2), np.uint8)
edges_mask = cv2.dilate(edges_mask, kernel, iterations=2)

# remove edeges from edges that are also in edges_mask
edges[edges_mask == 255] = 0

# Use Hough Line Transform to find line segments
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=20)

# Draw lines on a grayscale copy of the original image
image_with_lines =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image_with_lines, (x1, y1), (x2, y2), 255, 2)

# set all pixels that are not white to black
image_with_lines[image_with_lines < 255] = 0

# Use Hough Line Transform to find line segments on the image with lines
lines = cv2.HoughLinesP(image_with_lines, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=20)

# Draw the lines on a new copy of the original image
image_with_lines2 = image.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image_with_lines2, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Canny Edges", edges)
cv2.imshow("Image", image)
cv2.imshow("Image with lines", image_with_lines)
cv2.imshow("Image with lines2", image_with_lines2)

cv2.waitKey(0)
cv2.destroyAllWindows()