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
HOME = [5.5, 2.6, 0.1, 0.0, 0.0, np.sqrt(2), -np.sqrt(2)]
DESERT_TEST = [0.5, -1.0, 0.1, 0.0, 0.0, np.sqrt(2), np.sqrt(2)]
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
IMAGE_DEPTH = 3

# IMAGE MANIPULATION CONSTANTS
CROP_AMOUNT = 250
LOSS_FACTOR = 250
LOWER_PINK = np.array([200, 0, 200], dtype=np.uint8)
UPPER_PINK = np.array([255, 150, 255], dtype=np.uint8)
PINK_THRESHOLD = 10000
LOWER_RED = np.array([0, 0, 200], dtype=np.uint8)
UPPER_RED = np.array([100, 100, 255], dtype=np.uint8)
RED_THRESHOLD = 10000
LOWER_ROAD = np.array([0, 0, 78], dtype=np.uint8)
UPPER_ROAD = np.array([125, 135, 255], dtype=np.uint8)
LOWER_DIRT = np.array([130, 165, 170], dtype=np.uint8)
UPPER_DIRT = np.array([160, 210, 210], dtype=np.uint8)

# PID CONSTANTS
KP = 0.017
KD = 0.003
MAX_SPEED = 0.3
SPEED_DROP = 0.00055


# Some random states, will update later as needed for example, may have
# multiple driving states (depending on location), a pedestrian stop state,
# a clue state, etc.
class State:

    class Location(Enum):
        ROAD = 0
        DESERT = 1
        OFFROAD = 2
        MOUNTAIN = 3

    class Action(Enum):
        DRIVE = 0
        CRY = 1
        RESPAWN = 2
        EXPLODE = 3

    # Make state bitfield, so that multiple states can be active at once
    DRIVING = 0b00000001
    PINK = 0b00000010
    RED = 0b00000100
    CLUE = 0b00001000
    PINK_ON = 0b00010000

    # location list
    LOCATIONS = [Location.ROAD, Location.DESERT, Location.OFFROAD, Location.MOUNTAIN]

    def __init__(self):
        # Define the initial state
        self.location_count = 0
        self.current_location = self.LOCATIONS[self.location_count]
        self.last_state = self.DRIVING
        self.last_pink_time = 0
        self.current_state = self.DRIVING

    def choose_action(self):
        if self.current_state & self.PINK:
            if self.current_state & self.PINK_ON:
                self.last_pink_time = time.time()
                self.current_state &= ~self.PINK_ON
                self.location_count += 1
                self.current_location = self.LOCATIONS[self.location_count]
            elif time.time() - self.last_pink_time > 5:
                self.location_count += 1
                self.current_location = self.LOCATIONS[self.location_count]
                return self.Action.DRIVE
        elif self.current_state & self.RED:
            return self.Action.RESPAWN
        elif self.current_state & self.DRIVING:
            return self.Action.DRIVE
        # Based on the current state, choose an action to take
        # This will be where the priority of states is implemented
    

class topic_publisher:
    def __init__(self):
        self.running = False  # Prevent callback from running before competition starts
        self.bridge = CvBridge()
        self.state = State()
        self.previous_error = -100
        self.image_sub = rospy.Subscriber(
            "R1/pi_camera/image_raw", Image, self.callback
        )
        self.move_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=1)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_callback)
        self.last_image = np.zeros(
            (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), dtype=np.uint8
        )
        time.sleep(1)
        self.time_start = rospy.wait_for_message("/clock", Clock).clock.secs
        self.score_pub.publish("%s,%s,0,NA" % (TEAM_NAME, PASSWORD))
        self.running = True
        self.spawn_position(DESERT_TEST)

    def clock_callback(self, data):
        """Callback for the clock subscriber
        Publishes the score to the score tracker node when the competition ends
        """
        # if self.image_difference < 7000:
        #     self.spawn_position(HOME)
        #     print("Respawned")
        if self.running and data.clock.secs - self.time_start > END_TIME:
            self.running = False
            self.score_pub.publish("%s,%s,-1,NA" % (TEAM_NAME, PASSWORD))

    def callback(self, data):
        """Callback for the image subscriber
        Calls the appropriate function based on the state of the robot
        This is the main logic loop of the robot
        """
        cv_image = self.set_state(data)
        action = self.state.choose_action()

        # for debugging
        state_list = {0b00000001:"DRIVING", 0b00000010:"PINK", 0b00000100:"RED", 0b00001000:"CLUE", 0b00010000:"PINK_ON"}
        print(self.state.current_location)
        cv2.putText(
            cv_image,
            str(self.state.current_state),
            (1000, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if (action == State.Action.EXPLODE):  # TODO: shut down the script? - may not even need this state once this is implemented
            self.move_pub.publish(Twist())
            return

        # new_image = np.array(cv_image)
        # self.image_difference = cv2.norm(self.last_image, new_image, cv2.NORM_L2)

        if action == State.Action.DRIVE and self.state.current_location == State.Location.ROAD:
            self.driving(cv_image)
        elif action == State.Action.DRIVE and self.state.current_location == State.Location.OFFROAD:
            self.offroad_driving(cv_image)
        # elif action == State.Action.CRY:
        #     # TODO: implement pedestrian state
        #     self.driving(cv_image)
        # elif action == State.Action.RESPAWN:
        #     # TODO: implement desert state
        #     self.driving(cv_image)

        self.last_image = np.array(cv_image)

    def set_state(self, data):
        """Returns the current state of the robot, based on the image data
        Based on what the state will be, new data may be returned as well
        """
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Generating masks to check for certain colurs
        # Pink
        pink_mask = cv2.inRange(cv_image, LOWER_PINK, UPPER_PINK)
        pink_pixel_count = cv2.countNonZero(pink_mask)
        print("pink pixel count:", pink_pixel_count)
        # Red
        red_mask = cv2.inRange(cv_image, LOWER_RED, UPPER_RED)
        red_pixel_count = cv2.countNonZero(red_mask)
        print("red pixel count:", red_pixel_count)

        # Create a local state that will be added by bitwise OR
        state = 0b00000000

        if not self.running:
            return -1, None
        
        if red_pixel_count > RED_THRESHOLD:
            state |= self.state.RED

        if pink_pixel_count > PINK_THRESHOLD:
            state |= self.state.PINK
            if (self.state.last_state & self.state.PINK) == 0:
                state |= self.state.PINK_ON
                
        state |= self.state.DRIVING

        self.state.last_state = self.state.current_state
        self.state.current_state = state
        
        print("state:", self.state.current_state)
        return cv_image

    def driving(self, cv_image):
        """Function for the DRIVING state"""
        contour_colour = (0, 255, 0)
        lost_left = False

        mask = cv2.inRange(cv_image, LOWER_ROAD, UPPER_ROAD)

        mask = mask[IMAGE_HEIGHT - CROP_AMOUNT :, :]

        # find where the road x position is
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        error = 0
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            # check if any point is at [0][0]

            if largest_contour[0][0][0] == 0 and largest_contour[0][0][1] == 0:
                lost_left = True

            else:
                M = cv2.moments(largest_contour)
                if M["m00"] == 0:
                    # print(contours)
                    return
                centroid_x = int(M["m10"] / M["m00"])
                y_offset = IMAGE_HEIGHT - CROP_AMOUNT
                largest_contour[:, 0, 1] += y_offset
                # draw contour onto image
                cv2.drawContours(cv_image, [largest_contour], -1, contour_colour, 2)
                # calculate the error
                error = (
                    centroid_x - IMAGE_WIDTH / 2
                )  # positive means the car should turn right

        else:  # no contours detected, so set error directly
            error = -IMAGE_WIDTH / 2 if self.previous_error < 0 else IMAGE_WIDTH / 2

        if lost_left:  # This will bias the car to the left if the left side is lost
            error = -LOSS_FACTOR
            # write onto image
            cv2.putText(
                cv_image,
                "Lost LEFT",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Driving Image", cv_image)
        cv2.imshow("Mask", mask)
        cv2.waitKey(3)

        # PID controller
        move = Twist()

        print("Error", error)
        derivative = error - self.previous_error
        self.previous_error = error

        move.angular.z = -(KP * error + KD * derivative)
        print("angular speed:", move.angular.z, "\n")

        # decrease linear speed as angular speed increases from a max of 3 down to 1.xx? if abs(error) > 400
        move.linear.x = max(0, MAX_SPEED - SPEED_DROP * abs(error))

        self.move_pub.publish(move)
    def offroad_driving(self, cv_image):
        print("desert driving")
        lost_left = False
        contour_colour = (0, 255, 0)


        mask = cv2.inRange(cv_image, LOWER_DIRT, UPPER_DIRT)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(cv_image, [contour], -1, contour_colour, 2)
        
        cv2.imshow("Desert Driving Image", cv_image)
        cv2.waitKey(3)
        move = Twist()
        move.linear.x = 0.1
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

    def pedestrian(self, cv_image):
        """Function for the PEDESTRIAN state"""
        start_time = time.time()
        move = Twist()
        move.angular.z = 0.0
        move.linear.x = 0.1
        while time.time() - start_time < 3:
            self.move_pub.publish(move)


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
