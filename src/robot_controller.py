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
END_TIME = 5000000
HOME = [5.5, 2.0, 0.1, 0.0, 0.0, np.sqrt(2), -np.sqrt(2)]
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
IMAGE_DEPTH = 3

#CONSTANTS
CROP_AMOUNT = 250
LOSS_FACTOR = 1
LOWER_PINK = np.array([200, 0, 200], dtype=np.uint8)
UPPER_PINK = np.array([255, 150, 255], dtype=np.uint8)
LOWER_RED = np.array([0, 0, 200], dtype=np.uint8)
UPPER_RED = np.array([100, 100, 255], dtype=np.uint8)
LOWER_COLOR = np.array([0, 0, 78], dtype=np.uint8)
UPPER_COLOR = np.array([125, 135, 255], dtype=np.uint8)


# Some random states, will update later as needed for example, may have
# multiple driving states (depending on location), a pedestrian stop state,
# a clue state, etc.
class State(Enum):
    SHUTDOWN = 1
    DRIVING = 2
    ACTIVE = 3
    ENDED = 4
    PEDESTRIAN = 5


class topic_publisher:
    def __init__(self):
        self.bridge = CvBridge()
        self.previous_error = -100
        self.image_sub = rospy.Subscriber(
            "R1/pi_camera/image_raw", Image, self.callback
        )
        self.move_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=1)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        self.clock_sub = rospy.Subscriber(
            "/clock", Clock, self.clock_callback, queue_size=1
        )
        self.running = False  # Prevent callback from running before competition starts
        self.image_difference = 100000
        self.last_image = np.zeros(
            (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), dtype=np.uint8
        )
        print("before sleep", time.time())
        time.sleep(1)
        print("after sleep, before wait", time.time())
        self.time_start = rospy.wait_for_message("/clock", Clock).clock.secs
        print("after wait", time.time())
        self.score_pub.publish("%s,%s,0,NA" % (TEAM_NAME, PASSWORD))
        self.running = True
        print("init done", time.time())
        self.spawn_position(HOME)

    def clock_callback(self, data):
        """Callback for the clock subscriber
        Publishes the score to the score tracker node when the competition ends
        """
        # if self.image_difference < 7000:
        #     self.spawn_position(HOME)
        #     print("Respawned")
        print("Clock")
        if self.running and data.clock.secs - self.time_start > END_TIME:
            self.running = False
            self.score_pub.publish("%s,%s,-1,NA" % (TEAM_NAME, PASSWORD))

    def callback(self, data):
        """Callback for the image subscriber
        Calls the appropriate function based on the state of the robot
        This is the main logic loop of the robot
        """
        state, new_data = self.get_state(data)

        new_image = np.array(self.bridge.imgmsg_to_cv2(new_data, "bgr8"))
        self.image_difference = cv2.norm(self.last_image, new_image, cv2.NORM_L2)

        if (
            state == State.SHUTDOWN
        ):  # TODO: shut down the script? - may not even need this state once this is implemented
            self.move_pub.publish(Twist())
        elif state == State.DRIVING:
            self.driving(new_data)
        elif state == State.PEDESTRIAN:
            self.pedestrian(new_data)

        self.last_image = np.array(self.bridge.imgmsg_to_cv2(new_data, "bgr8"))

    def get_state(self, data):
        """Returns the current state of the robot, based on the image data
        Based on what the state will be, new data may be returned as well
        """
        # Depending on needs of states, the state datatype may have to be changed to allow for multiple states at once

        if (not self.running):  # if the competition is over, stop the robot, do nothing else
            return State.SHUTDOWN, data
        else:
            return State.DRIVING, data

    def driving(self, data):
        """Function for the DRIVING state"""

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        height, width = cv_image.shape[:2]
        
        contour_colour = (0, 255, 0)
        lost = False
        lost_left = False
        lost_right = False

        pink_mask = cv2.inRange(cv_image, LOWER_PINK, UPPER_PINK)
        pink_pixel_count = cv2.countNonZero(pink_mask)
        print("pink pixel count:", pink_pixel_count)

        red_mask = cv2.inRange(cv_image, LOWER_RED, UPPER_RED)
        red_pixel_count = cv2.countNonZero(red_mask)
        print("red pixel count:", red_pixel_count)
        # if red_pixel_count > 500:
        #     TODO: Move this type of check into get_state()
        #     self.state = State.PEDESTRIAN
        #     return

        masktest = cv2.inRange(cv_image, LOWER_COLOR, UPPER_COLOR)

        # image processing, including cropping and thresholding
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, black_image = cv2.threshold(gray_image, 100, 255, type=cv2.THRESH_BINARY)
        black_image = cv2.bitwise_not(black_image)
        cropped_image = black_image[height - CROP_AMOUNT :, :]
        _, mask = cv2.threshold(cropped_image, 128, 255, cv2.THRESH_BINARY)

        mask = masktest[height - CROP_AMOUNT :, :]

        # find where the road x position is
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        error = 0
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            bottom_left, top_left, bottom_right, top_right = self.find_corner_points(
                largest_contour
            )

            cv2.circle(
                cv_image,
                (bottom_left[0][0], bottom_left[0][1] + (height - CROP_AMOUNT)),
                7,
                (0, 0, 255),
                -1,
            )
            cv2.circle(
                cv_image,
                (top_left[0][0], top_left[0][1] + (height - CROP_AMOUNT)),
                7,
                (0, 0, 0),
                -1,
            )
            cv2.circle(
                cv_image,
                (bottom_right[0][0], bottom_right[0][1] + (height - CROP_AMOUNT)),
                7,
                (255, 0, 0),
                -1,
            )
            cv2.circle(
                cv_image,
                (top_right[0][0], top_right[0][1] + (height - CROP_AMOUNT)),
                7,
                (0, 255, 0),
                -1,
            )

            for point in largest_contour:
                cv2.circle(
                    cv_image, (point[0][0], point[0][1]), 10, (255, 255, 255), -1
                )

            if bottom_left[0][0] == top_left[0][0]:
                lost_left = True
                contour_colour = (0, 255, 255)
            if bottom_right[0][0] == top_right[0][0]:
                lost_right = True
                contour_colour = (0, 165, 255)
            if lost_left and lost_right:
                lost = True
                contour_colour = (0, 0, 255)
            print("lost:", lost, "lost left:", lost_left, "lost right:", lost_right)

            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                # print(contours)
                return
            centroid_x = int(M["m10"] / M["m00"])
            # draw circle onto image
            cv2.circle(
                cv_image, (centroid_x, height - CROP_AMOUNT // 2), 5, (0, 0, 255), -1
            )
            # shift contour points to account for cropping
            y_offset = height - CROP_AMOUNT
            largest_contour[:, 0, 1] += y_offset
            # draw contour onto image
            cv2.drawContours(cv_image, [largest_contour], -1, contour_colour, 2)
            # calculate the error
            error = centroid_x - width / 2  # positive means the car should turn left

            if lost_left:
                error = abs(self.previous_error) + LOSS_FACTOR
            if lost_right:
                error = -abs(self.previous_error) - LOSS_FACTOR
            if lost:
                error = (
                    self.previous_error - LOSS_FACTOR
                    if self.previous_error < 0
                    else self.previous_error + LOSS_FACTOR
                )
        else:  # no contours detected, so set error directly
            error = -width / 2 if self.previous_error < 0 else width / 2

        cv2.imshow("Driving Image", cv_image)
        cv2.imshow("Road Mask Test", mask)
        cv2.waitKey(3)

        # PID controller
        move = Twist()

        print("Error", error)
        Kp = 0.017
        Kd = 0.003
        derivative = error - self.previous_error
        self.previous_error = error

        move.angular.z = -(Kp * error + Kd * derivative)
        print("angular speed:", move.angular.z, "\n")
        # decrease linear speed as angular speed increases from a max of 3 down to 1.xx? if abs(error) > 400
        move.linear.x = max(0, 0.1 - 0.00065 * abs(error))
        # move.linear.x = 0.2

        self.move_pub.publish(move)

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

    def find_corner_points(self, contour):
        """Finds the corner points of a contour
        Returns a list of the corner points
        """
        top_left = min(
            filter(
                lambda p: p[0][1] == min(contour, key=lambda x: x[0][1])[0][1], contour
            ),
            key=lambda x: x[0][0],
        )
        bottom_left = min(
            filter(
                lambda p: p[0][1] == max(contour, key=lambda x: x[0][1])[0][1], contour
            ),
            key=lambda x: x[0][0],
        )
        bottom_right = max(contour, key=lambda point: (point[0][0] + point[0][1]))
        top_right = max(
            filter(
                lambda p: p[0][1] == min(contour, key=lambda x: x[0][1])[0][1], contour
            ),
            key=lambda x: x[0][0],
        )

        return bottom_left, top_left, bottom_right, top_right

    def pedestrian(self, data):
        """Function for the PEDESTRIAN state"""
        start_time = time.time()
        move = Twist()
        move.angular.z = 0.0
        move.linear.x = 0.1
        while time.time() - start_time < 3:
            self.move_pub.publish(move)
        self.state = State.DRIVING

def main(args):
    print("main")
    rospy.init_node("topic_publisher")
    topic_publisher()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
    print("main done")


if __name__ == "__main__":
    main(sys.argv)
