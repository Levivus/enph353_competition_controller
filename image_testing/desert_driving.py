import cv2
import numpy as np
import os
"""Desert driving script
This is used to test the performance of the car in the desert environment.
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(script_dir, 'desert_images', 'image_6.png'))

def process_image(image):
    lower_tunnel = np.array([80, 100, 185])
    upper_tunnel = np.array([95, 120, 195])

    tunnel_mask = cv2.inRange(image, lower_tunnel, upper_tunnel)
    tunnel_mask = cv2.dilate(tunnel_mask, np.ones((2,2),np.uint8),iterations = 3)
    image[tunnel_mask > 0] = 0

    

    return image, tunnel_mask

cv2.imshow('image', image)
img, mask = process_image(image)
cv2.imshow('mask', mask)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()