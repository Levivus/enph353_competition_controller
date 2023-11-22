#! /usr/bin/env python3

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

class topic_publisher:

  def __init__(self):
    print("init")
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("R1/pi_camera/image_raw",Image,self.callback)
    self.clock_sub = rospy.Subscriber("/clock", int)
    self.move_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=1)
    self.score_pub = rospy.Publisher("/score_tracker", std_msgs/String, queue_size=1)
    self.previous_error = -100
    print("init done")

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    height, width = cv_image.shape[:2]

    # #image processing, including cropping and thresholding
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, black_image = cv2.threshold(gray_image, 100, 255, type=cv2.THRESH_BINARY)
    black_image = cv2.bitwise_not(black_image)
    cropped_image = black_image[height-30:, :]

    # #find where the road x position is
    _, mask = cv2.threshold(cropped_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    error = 0
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
          print(contours)
          return
        centroid_x = int(M["m10"] / M["m00"])
        #draw circle onto image
        cv2.circle(cv_image, (centroid_x, height-15), 5, (0, 0, 255), -1)
        #shift contour points to account for cropping
        y_offset = height - 30
        largest_contour[:, 0, 1] += y_offset
        #draw contour onto image
        cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 0), 2)
        #calculate the error
        error = centroid_x - width/2  #positive means the car should turn left
    else: #no contours detected, so set error directly
        error = -width/2 if self.previous_error < 0 else width/2

    

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    # #PID controller
    move = Twist()

    Kp = 0.017
    Kd = 0.03
    derivative = error - self.previous_error
    self.previous_error = error

    move.angular.z = -(Kp * error + Kd * derivative)
    #decrease linear speed as angular speed increases from a max of 3 down to 1.xx? if abs(error) > 400
    move.linear.x = max(0, 3 - 0.0065 * abs(error))

    self.image_pub.publish(move)

def main(args): 
  tp = topic_publisher()
  print("main")
  rospy.init_node('topic_publisher')
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()
  print("main done")

if __name__ == '__main__':
    main(sys.argv)
