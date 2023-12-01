import cv2
import numpy as np
import os

LOWER_DIRT = np.array([140, 164, 168], dtype=np.uint8)
UPPER_DIRT = np.array([186, 220, 228], dtype=np.uint8)

script_dir = os.path.dirname(os.path.abspath(__file__))
# Specify the relative path to the image in the driving_images folder
image_path = os.path.join(script_dir, 'driving_images', 'image_330.png')


def process_image(image):
    # Read the image

    mask = cv2.inRange(image, LOWER_DIRT, UPPER_DIRT)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Define a kernel (structuring element)
    kernel = np.ones((3, 3), np.uint8)

    dilated_image = cv2.dilate(mask, kernel, iterations=1)

    # Perform erosion
    eroded_image = cv2.erode(dilated_image, kernel, iterations=2)
    # cv2.imshow('Eroded Image', eroded_image)

    # Perform dilation
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=8)

    dilate_contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dilate_contours = sorted(dilate_contours, key=cv2.contourArea, reverse=True)

    top_two_contours = dilate_contours[:2]

    return top_two_contours

for i in range(0, 1381, 10):
    image_path = os.path.join(script_dir, 'driving_images', 'image_' + str(i) + '.png')
    image = cv2.imread(image_path)
    top_two_contours = process_image(image)
    cv2.drawContours(image, top_two_contours, -1, (0, 255, 0), 3)
    cv2.imshow('Original Image', image)
    # cv2.imshow('New Contours', image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()