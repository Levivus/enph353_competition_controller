#! /usr/bin/env python3

import sys
import rospy
import cv2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import threading

class topic_publisher:

  def __init__(self):
    self.bridge = CvBridge()
    # self.previous_error = -100
    # self.image_sub = rospy.Subscriber("R1/pi_camera/image_raw",Image,self.callback)
    # self.move_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=1)
    # self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
    # self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_callback, queue_size=1)
    # self.running = False # Prevent callback from running before competition starts
    # time.sleep(1)
    # self.time_start = rospy.wait_for_message("/clock", Clock).clock.secs
    # self.score_pub.publish("%s,%s,0,NA" % (TEAM_NAME, PASSWORD))
    # self.running = True
  
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

  def keyboard_input_handler(self):
    while not rospy.is_shutdown():
        # Get user input from the keyboard
        user_input = input("Press 'r' to spawn position, 'q' to quit: ")

        # Check the user input
        if user_input.lower() == 'r':
            # Call the spawn_position function with your desired position
            self.spawn_position([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif user_input.lower() == 'q':
            rospy.signal_shutdown("User requested shutdown.")
  

def main(args): 
  # print("main")
  rospy.init_node('topic_publisher')
  topic_publisher()
  # Start a new thread to handle keyboard input
  input_thread = threading.Thread(target=topic_publisher.keyboard_input_handler, args=())
  input_thread.start()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()
  print("main done")

if __name__ == '__main__':
    main(sys.argv)
