#! /usr/bin/env python3

import sys
import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler
import threading
import time


class topic_publisher:
    def __init__(self):
        self.running = False  # Prevent callback from running before competition starts
        time.sleep(1)
        self.running = True

    def spawn_position(self, position):
        msg = ModelState()
        msg.model_name = "R1"

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = position[3]
        msg.pose.orientation.y = position[4]
        msg.pose.orientation.z = position[5]
        msg.pose.orientation.w = position[6]

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            resp = set_state(msg)

        except rospy.ServiceException:
            print("Service call failed")

    def keyboard_input_handler(self):
        while not rospy.is_shutdown():
            # Get user input from the keyboard
            user_input = input(
                "'h' for the start, 'p' for the first pink, 't' for behind the first pink line, 'r' for just behind the red line, followed by the orientation in degrees (00), 'q' to quit: "
            )
            orientation = float(user_input[1:])
            location = user_input[0]
            print("orientation: ", orientation)
            x, y, z, w = quaternion_from_euler(*(0.0, 0.0, np.radians(orientation)))
            print("x, y, z, w: ", x, y, z, w)

            # Check the user input
            if location == "h":
                self.spawn_position([5.5, 2.6, 0.1, x, y, z, w])
            elif location == "r":
                self.spawn_position([4.5, 1.0, 0.1, x, y, z, w])
            elif location == "p":
                self.spawn_position([0.5, 0.0, 0.1, x, y, z, w])
            elif location == "t":
                self.spawn_position([0.5, -1.0, 0.1, x, y, z, w])
            elif user_input.lower() == "q":
                rospy.signal_shutdown("User requested shutdown.")


def main(args):
    # print("main")
    rospy.init_node("topic_publisher")
    tp = topic_publisher()
    # Start a new thread to handle keyboard input
    input_thread = threading.Thread(target=tp.keyboard_input_handler, args=())
    input_thread.start()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    print("main done")


if __name__ == "__main__":
    main(sys.argv)
