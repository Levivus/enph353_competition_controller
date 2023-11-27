#! /usr/bin/env python3

import sys
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from enum import Enum
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import numpy as np

TEAM_NAME = "MchnEarn"
PASSWORD = "pswd"
END_TIME = 5

class State(Enum):
    SHUTDOWN = 1 # Some random states, will update later as needed
    DRIVING = 2  # for example, may have multiple driving states (depending on location),
    ACTIVE = 3   # a pedestrian stop state, a clue state, etc.
    ENDED = 4

class topic_publisher:

  def __init__(self):
    self.bridge = CvBridge()
    self.previous_error = -100
    self.image_sub = rospy.Subscriber("R1/pi_camera/image_raw",Image,self.callback)
    self.move_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=1)
    self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
    self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_callback, queue_size=1)
    self.running = False # Prevent callback from running before competition starts
    self.last_image_time = rospy.get_rostime()
    self.respawn_time = rospy.Timer(rospy.Duration(0.5), self.respawn_callback)
    time.sleep(1)
    self.time_start = rospy.wait_for_message("/clock", Clock).clock.secs
    self.score_pub.publish("%s,%s,0,NA" % (TEAM_NAME, PASSWORD))
    self.running = True


  def clock_callback(self, data):
    """Callback for the clock subscriber
    Publishes the score to the score tracker node when the competition ends
    """
    if self.running and data.clock.secs - self.time_start > END_TIME:
      self.running = False
      self.score_pub.publish("%s,%s,-1,NA" % (TEAM_NAME, PASSWORD))
    

  def callback(self,data):
    """Callback for the image subscriber
    Calls the appropriate function based on the state of the robot
    This is the main logic loop of the robot
    """
    state, new_data = self.get_state(data)
    if state == State.SHUTDOWN: #TODO: shut down the script? - may not even need this state once this is implemented
      self.move_pub.publish(Twist())
    elif state == State.DRIVING:
      self.driving(new_data)



  def get_state(self, data):
    """Returns the current state of the robot, based on the image data
    Based on what the state will be, new data may be returned as well
    """
    # Depending on needs of states, the state datatype may have to be changed to allow for multiple states at once

    if not self.running: #if the competition is over, stop the robot, do nothing else
      return State.SHUTDOWN, data
    
    return State.DRIVING, data
  
  
  def driving(self, data):
    """Function for the DRIVING state"""

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    self.respawn_time = rospy.get_rostime()

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
          # print(contours)
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

    Kp = 0.0017
    Kd = 0.003
    derivative = error - self.previous_error
    self.previous_error = error

    move.angular.z = -(Kp * error + Kd * derivative)
    #decrease linear speed as angular speed increases from a max of 3 down to 1.xx? if abs(error) > 400
    move.linear.x = max(0, 0.05 - 0.00065 * abs(error))
    move.linear.x = 0.2

    self.move_pub.publish(move)
  
  def respawn_callback(self, event):
    """Callback for the respawn timer
    Respawns the robot if it has been stuck for too long
    """
    if rospy.get_rostime() - self.last_image_time > rospy.Duration(3):
      self.spawn_position([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  def spawn_position(self, position):

    msg = ModelState()
    msg.model_name = 'R1'

    msg.pose.position.x = position[0]
    msg.pose.position.y = position[1]
    msg.pose.position.z = position[2]
    msg.pose.orientation.x = position[3]
    msg.pose.orientation.y = position[4]
    msg.pose.orientation.z = position[5]
    msg.pose.orientation.w = position[6]

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( msg )

    except rospy.ServiceException:
        print ("Service call failed")
  

def main(args): 
  # print("main")
  rospy.init_node('topic_publisher')
  topic_publisher()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()
  print("main done")

if __name__ == '__main__':
    main(sys.argv)
