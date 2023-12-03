import cv2
import numpy as np
import os
"""Obstacle detection function
This is used to detect obsatcles, which includes the pedestrian, the car adn baby yoda.
It uses ...
"""

bg_subtractor = cv2.createBackgroundSubtractorMOG2()


# write a function which detects an obstacle in front of the robot and returns true if there is an obstacle and false if there is no obstacle
def obstacle_detection(image):
    # get size of image
    height, width, _ = image.shape

    # crop image to middle 1/2
    image = image[int(height/2):, ]
    kernel = np.ones((5, 5), np.uint8)
    # make all pixels close to white red
    mask = cv2.inRange(image, (250, 250, 250), (255, 255, 255))
    image[mask>0] = [82,82,82]
    image = cv2.erode(image, kernel, iterations=4)
    cv2.imshow('mask', image)
    mask = cv2.inRange(image, (70, 70, 70), (85, 85, 85))
    # erode to clean it up
    mask = cv2.erode(mask, kernel, iterations=4)
    cv2.imshow('erode', mask)
    mask = cv2.dilate(mask, kernel, iterations=3)
    cv2.imshow('e+d', mask)

    # # make it grayscale and show it
    # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('gray', grey)

    # mask = cv2.inRange(grey, 255, 255)
    # grey[mask>0] = [82]

    # grey = cv2.erode(grey, kernel, iterations=4)
    # grey = cv2.dilate(grey, kernel, iterations=3)
    # # cv2.imshow('erode 2', grey)

    # # make it hsv and show it
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # cv2.imshow('hsv', hsv)

    # mask = cv2.inRange(hsv, (0, 0, 250), (0, 0, 255))
    # hsv[mask>0] = [0,0,82]

    # hsv = cv2.erode(hsv, kernel, iterations=4)
    # hsv = cv2.dilate(hsv, kernel, iterations=3)
    # cv2.imshow('erode 3', hsv)

    # find the largest contour and draw it on the original image filled din with white
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = 0.01 * cv2.arcLength(max(contours, key=cv2.contourArea), True)
    approx_polygon = cv2.approxPolyDP(max(contours, key=cv2.contourArea), epsilon, True)
    cv2.drawContours(image, [approx_polygon], -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.imshow('contours', image)

    # Display the original image, binary mask, and foreground

    mask = cv2.inRange(image, (80, 80, 80), (90, 90, 90))
    kernel = np.ones((2, 2), np.uint8)

    # cv2.imshow('Mask', mask)

    # mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours in the dilated edges
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = 0.01 * cv2.arcLength(max(contours, key=cv2.contourArea), True)
    approx_polygon = cv2.approxPolyDP(max(contours, key=cv2.contourArea), epsilon, True)

    # Define the color range bands to exclude (for example, excluding red tones)
    lower_band1 = np.array([80, 80, 80])
    upper_band2 = np.array([90, 90, 90])
    lower_band2 = np.array([250, 250, 250])
    upper_band2 = np.array([255, 255, 255])
    lower_band3 = np.array([65, 135, 25])
    upper_band3 = np.array([73, 143, 33])
    lower_band4 = np.array([0, 0, 250])
    upper_band4 = np.array([5, 5, 255])

    # Create a mask for the specified color range
    color_mask1 = cv2.inRange(image, lower_band1, upper_band2)
    image[color_mask1>0] = [255,255,255]
    color_mask2 = cv2.inRange(image, lower_band2, upper_band2)
    image[color_mask2>0] = [255,255,255]
    color_mask3 = cv2.inRange(image, lower_band3, upper_band3)
    image[color_mask3>0] = [255,255,255]
    color_mask4 = cv2.inRange(image, lower_band4, upper_band4)
    image[color_mask4>0] = [255,255,255]

    # dialte to clean it up
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)

    # Check if any pixel falls outside the specified color range
    if cv2.countNonZero(color_mask1) > 100:
        print("yay")

    # Draw the contours on the mask
    cv2.drawContours(image, [approx_polygon], -1, (0, 0, 255), thickness=2)

    # Display the original image and the contour mask
    # cv2.imshow('Original Image', image)
    # cv2.imshow('mask', color_mask1)
    # cv2.imshow('mask2', color_mask2)
    # cv2.imshow('mask3', color_mask3)
    # cv2.imshow('mask4', color_mask4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

obstacle_detection(cv2.imread('/home/fizzer/ros_ws/src/enph353_competition_controller/image_testing/obstacle_images/image_51.png'))