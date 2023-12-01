#!/usr/bin/env python

import numpy as np
import cv2 as cv
import time

img = cv.imread('test1dirt.png',cv.IMREAD_COLOR)
# img = cv.imread('test2.png',cv.IMREAD_COLOR)
# img = cv.imread('test3.png',cv.IMREAD_COLOR)

# img = cv.medianBlur(img,5)

# Convert BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

ub = 186
ug = 220
ur = 228
lb = 140
lg = 164
lr = 168
lower_bgr = np.array([lb,lg,lr])
upper_bgr = np.array([ub,ug,ur])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(img, lower_bgr, upper_bgr)
window_name = "BGR Calibrator"
cv.namedWindow(window_name)

def nothing(x):
    print("Trackbar value: " + str(x))
    pass

# create trackbars for Upper HSV
cv.createTrackbar('UpperB',window_name,0,255,nothing)
cv.setTrackbarPos('UpperB',window_name, ub)

cv.createTrackbar('UpperG',window_name,0,255,nothing)
cv.setTrackbarPos('UpperG',window_name, ug)

cv.createTrackbar('UpperR',window_name,0,255,nothing)
cv.setTrackbarPos('UpperR',window_name, ur)

# create trackbars for Lower HSV
cv.createTrackbar('LowerB',window_name,0,255,nothing)
cv.setTrackbarPos('LowerB',window_name, lb)

cv.createTrackbar('LowerG',window_name,0,255,nothing)
cv.setTrackbarPos('LowerG',window_name, lg)

cv.createTrackbar('LowerR',window_name,0,255,nothing)
cv.setTrackbarPos('LowerR',window_name, lr)

font = cv.FONT_HERSHEY_SIMPLEX

print("Loaded images")

while(1):
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(img, lower_bgr, upper_bgr)
    cv.putText(mask,'Lower BGR: [' + str(lb) +',' + str(lg) + ',' + str(lr) + ']', (10,30), font, 0.5, (200,255,155), 1, cv.LINE_AA)
    cv.putText(mask,'Upper BGR: [' + str(ub) +',' + str(ug) + ',' + str(ur) + ']', (10,60), font, 0.5, (200,255,155), 1, cv.LINE_AA)

    cv.imshow(window_name,mask)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of Upper HSV trackbars
    ub = cv.getTrackbarPos('UpperB',window_name)
    ug = cv.getTrackbarPos('UpperG',window_name)
    ur = cv.getTrackbarPos('UpperR',window_name)
    # get current positions of Lower HSCV trackbars
    lb = cv.getTrackbarPos('LowerB',window_name)
    lg = cv.getTrackbarPos('LowerG',window_name)
    lr = cv.getTrackbarPos('LowerR',window_name)
    upper_bgr = np.array([ub,ug,ur])
    lower_bgr = np.array([lb,lg,lr])

    # save the image when i hit a buttom that says save
    if k == ord('s'):
        cv.imwrite('hsv_threshold.png',mask)
        break

    time.sleep(.1)

cv.destroyAllWindows()