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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import Levenshtein

import os
import os.path
from os import path
import inspect

import tensorflow as tf


TEAM_NAME = "MchnEarn"
PASSWORD = "pswd"
END_TIME = 230
HOME = [5.5, 2.6, 0.1, 0.0, 0.0, np.sqrt(2), -np.sqrt(2)]
OFFROAD_TEST = [0.5, -0.5, 0.1, 0.0, 0.0, np.sqrt(2), np.sqrt(2)]
MOUNTAIN_TEST = [-4.0, -2.35, 0.1, 0.0, 0.0, 0.0, 0.0]
FIRST_PINK = [0.5, 0.0, 0.1, 0.0, 0.0, np.sqrt(2), np.sqrt(2)]
DESERT_TEST = [-1.5, 1.4, 0.1, 0.0, 0.0, 1, 0]
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
IMAGE_DEPTH = 3
OBSTACLE_FRAME_COUNT = 10
OBSTACLE_THRESHOLD = 500

# IMAGE MANIPULATION CONSTANTS
CROP_AMOUNT = 250
LOSS_FACTOR = 200
RESPAWN_THRESHOLD = 5
LOWER_PINK = np.array([200, 0, 200], dtype=np.uint8)
UPPER_PINK = np.array([255, 150, 255], dtype=np.uint8)
PINK_THRESHOLD = 30000
DESERT_PINK_THRESHOLD = 8000
LOWER_RED = np.array([0, 0, 200], dtype=np.uint8)
UPPER_RED = np.array([100, 100, 255], dtype=np.uint8)
RED_THRESHOLD = 60000
LOWER_ROAD = np.array([0, 0, 78], dtype=np.uint8)
UPPER_ROAD = np.array([125, 135, 255], dtype=np.uint8)
LOWER_DIRT = np.array([140, 164, 168], dtype=np.uint8)
UPPER_DIRT = np.array([186, 220, 228], dtype=np.uint8)
FLOOR_LOWER_HSV = np.array([6, 23, 118])
FLOOR_UPPER_HSV = np.array([84, 255, 255])
GREEN_LOWER = np.array([60, 136, 25])
GREEN_UPPER = np.array([75, 145, 35])
KERNEL_2 = np.ones((2, 2), np.uint8)
KERNEL_3 = np.ones((3, 3), np.uint8)
KERNEL_5 = np.ones((5, 5), np.uint8)

# PID CONSTANTS
KP = 0.02
KD = 0.004
OKP = 0.45  # desert KP, multiplies KP
OKD = 0.7  # desert KD, multiplies KD
OKX = 1  # desert lateral multiplierl
OKY = 0.25  # desert angle multiplier
MAX_SPEED = 0.8
SPEED_DROP = 0.00055
SPEED_DROP_OFFROAD = 0.0023
SPEED_DROP_MOUNTAIN = 0.0028
Y_MULT_CUTOFF = IMAGE_HEIGHT - 295
Y_MULT_CUTOFF_MOUNTAIN = IMAGE_HEIGHT - 300

CLUE_TYPES = {
    "SIZE": 1,
    "VICTIM": 2,
    "CRIME": 3,
    "TIME": 4,
    "PLACE": 5,
    "MOTIVE": 6,
    "WEAPON": 7,
    "BANDIT": 8,
}

CAPTURE_PATH = "/home/fizzer/ros_ws/src/enph353_competition_controller/image_testing/"
ABSOLUTE_PATH = "/home/fizzer/ros_ws/src/enph353_competition_controller/src/"


# Some random states, will update later as needed for example, may have
# multiple driving states (depending on location), a pedestrian stop state,
# a clue state, etc.
class State:
    class Location(Enum):
        ROAD = 0
        OFFROAD = 1
        DESERT = 2
        MOUNTAIN = 3

    # TODO: finaiilse these actions, delte the ones we dont use, etc.
    class Action(Enum):
        DRIVE = 0
        AVOID = 1
        RESPAWN = 2
        EXPLODE = 3
        SIT = 4

    # Make state bitfield, so that multiple states can be active at once
    NOTHING = 0b00000000
    DRIVING = 0b00000001
    PINK = 0b00000010
    OBSTACLE = 0b00000100
    CLUE = 0b00001000
    PINK_ON = 0b00010000

    # location list
    LOCATIONS = [Location.ROAD, Location.OFFROAD, Location.DESERT, Location.MOUNTAIN]

    def __init__(self):
        # Define the initial state
        self.location_count = 0
        self.current_location = self.LOCATIONS[self.location_count]
        self.tunnel_state = True
        self.turn_flag = False
        self.last_state = self.NOTHING
        self.last_pink_time = 0
        self.start_offroad_time = 500
        self.respawnTime = 0
        self.offroad_clue_found = False
        self.extra_mask_enabled = True
        self.never_on_top = True
        self.clue_first_seen = True
        self.first_mountain_call = True
        self.first_tunnel_line_found = False
        self.dark_brick_state = False
        self.clue_in_view = False
        self.clues_submitted = []
        self.best_clue = np.zeros(
            (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), dtype=np.uint8
        )
        # MAX INTEGER
        self.clue_improved_time = np.iinfo(np.int32).max
        self.current_time = 0
        self.max_area = 0
        self.current_state = self.NOTHING
        self.detection_enabled = True # CHANGE THIS LATER
        self.finished = False

    def set_location(self, location):
        self.current_location = location
        print("Location set to %s" % location)
        self.location_count = self.LOCATIONS.index(location)

    def choose_action(self):
        if self.finished:
            return self.Action.EXPLODE
        if self.current_state & self.PINK:
            if (
                self.current_state & self.PINK_ON
                and self.current_time - self.last_pink_time > 5
            ):
                self.detection_enabled = False
                self.last_pink_time = self.current_time
                self.current_state &= ~self.PINK_ON
                self.location_count += 1
                if self.location_count == 1:
                    self.start_offroad_time = self.current_time
                
                print("STATE CHANGED")
                self.current_location = self.LOCATIONS[self.location_count]
                return self.Action.DRIVE
        elif self.current_state & self.OBSTACLE:
            return self.Action.SIT
        elif self.current_state & self.DRIVING:
            return self.Action.DRIVE
        # Based on the current state, choose an action to take
        # This will be where the priority of states is implemented


class topic_publisher:
    def __init__(self):
        print("init")
        self.running = False  # Prevent callback from running before competition starts
        self.bridge = CvBridge()
        self.state = State()
        self.previous_error = 0
        self.previous_error_t = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.tunnel_time = False
        self.clue_centroid = None
        self.pink_tunnel_reached = 0

        self.image_sub = rospy.Subscriber(
            "R1/pi_camera/image_raw", Image, self.callback
        )
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        self.move_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=1)
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_callback)
        self.last_image = np.zeros(
            (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), dtype=np.uint8
        )
        time.sleep(1)
        self.score_pub.publish("%s,%s,0,NA" % (TEAM_NAME, PASSWORD))
        self.time_start = rospy.wait_for_message("/clock", Clock).clock.secs

        self.lite_model = tf.lite.Interpreter(
            model_path=ABSOLUTE_PATH + "quantized_model.tflite"
        )
        self.lite_model.allocate_tensors()
        self.input_details = self.lite_model.get_input_details()
        self.output_details = self.lite_model.get_output_details()

        self.running = True
        self.img_count = 0
        # self.spawn_position(DESERT_TEST)
        # self.state.set_location(State.Location.OFFROAD)
        print("init done")

    def clock_callback(self, data):
        """Callback for the clock subscriber
        Publishes the score to the score tracker node when the competition ends
        """
        self.state.current_time = data.clock.secs + data.clock.nsecs / 1000000000
        # If self.state.clues does not contain clues 5 6 and 7, and the current time is past 2:30, respawn
        # if (1 not in self.state.clues_submitted or 2 not in self.state.clues_submitted or 3 not in self.state.clues_submitted or 3 not in self.state.clues_submitted) and self.state.current_time > 150 and self.state.current_time - self.state.respawnTime > 30:
        #     print("RESPAWNING 1")
        #     self.spawn_position(HOME)
        #     self.state.current_state = self.state.DRIVING
        #     self.state.set_location(State.Location.ROAD)

        #     self.state.respawnTime = self.state.current_time


        # elif (5 not in self.state.clues_submitted or 6 not in self.state.clues_submitted or 7 not in self.state.clues_submitted) and self.state.current_time > 150 and self.state.current_time - self.state.respawnTime > 30:
        #     print("RESPAWNING 2")
        #     self.spawn_position(FIRST_PINK)
        #     self.state.current_state = self.state.DRIVING
        #     self.state.set_location(State.Location.OFFROAD)
        #     self.state.clues_submitted = []
        #     self.state.max_area = 0
        #     self.state.offroad_clue_found = False
        #     self.state.extra_mask_enabled = True
        #     self.state.never_on_top = True
        #     self.state.clue_first_seen = True
        #     self.state.first_mountain_call = True
        #     self.state.first_tunnel_line_found = False
        #     self.state.clue_in_view = False
        #     self.state.start_offroad_time = self.state.current_time

        if self.state.current_time - self.time_start > END_TIME:
            print("END TIME REACHED")
            self.score_pub.publish("%s,%s,-1,NA" % (TEAM_NAME, PASSWORD))
            self.running = False

    def callback(self, data):
        """Callback for the image subscriber
        Calls the appropriate function based on the state of the robot
        This is the main logic loop of the robot
        """

        cv_image = self.set_state(data)
        action = self.state.choose_action()

        if (
            action == State.Action.EXPLODE
        ):  # TODO: shut down the script? - may not even need this state once this is implemented
            # Turn left for 0.5 seconds
            move = Twist()
            move.angular.z = 0.3
            self.move_pub.publish(move)
            rospy.sleep(0.5)
            # Publish the score
            self.score_pub.publish("%s,%s,-1,NA" % (TEAM_NAME, PASSWORD))
            rospy.sleep(0.1)
            # Start moving forward
            move.angular.z = 0
            move.linear.x = 1.2
            self.move_pub.publish(move)
            # Shutdown rospy, and exit the script
            rospy.signal_shutdown("Finished")
            sys.exit()

            return

        if self.running:
            new_image = np.array(cv_image)
            self.image_difference = np.mean((new_image - self.last_image) ** 2)
            if (
                self.image_difference < RESPAWN_THRESHOLD
                and self.state.current_time - self.time_start > 10
            ):
                self.spawn_position(HOME)
                self.state.current_state = self.state.DRIVING
                self.state.set_location(State.Location.ROAD)
            self.last_image = new_image

        if (
            action == State.Action.DRIVE
            and self.state.current_location == State.Location.ROAD
        ):
            self.driving(cv_image)
        elif (
            action == State.Action.DRIVE
            and self.state.current_location == State.Location.OFFROAD
        ):
            self.offroad_driving(cv_image)
        elif (
            action == State.Action.DRIVE
            and self.state.current_location == State.Location.DESERT
        ):
            self.desert(cv_image)
        elif (
            action == State.Action.DRIVE
            and self.state.current_location == State.Location.MOUNTAIN
        ):
            self.mountain(cv_image)
        elif action == State.Action.SIT:
            self.avoid_obstacle()

    def set_state(self, data):
        """Returns the current state of the robot, based on the image data
        Based on what the state will be, new data may be returned as well
        """
        if not self.running:
            return -1, None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Check for clues and update the best clue state
        self.update_clue(cv_image)

        # If the clue has not improved in 1 seconds find the letters, and reset the max area
        if self.state.current_time - self.state.clue_improved_time > 2:
            print("Submitting clue")
            self.submit_clue()
            self.state.clue_improved_time = (
                self.state.current_time + 300
            )  # Set the time after comp, so it doesn't submit again
            self.state.max_area = 0

        # Generating masks to check for certain colurs
        # Pink
        pink_mask = cv2.inRange(cv_image, LOWER_PINK, UPPER_PINK)
        pink_pixel_count = cv2.countNonZero(pink_mask)

        # Create a local state that will be added by bitwise OR
        state = 0b00000000

        if self.obstacle_detection(cv_image) and pink_pixel_count < 1000:
            print("OBSTACLE DETECTED")
            state |= self.state.OBSTACLE

        if self.state.current_location == self.state.Location.OFFROAD:
            threshold = DESERT_PINK_THRESHOLD
        elif self.state.current_location == self.state.Location.ROAD:
            threshold = PINK_THRESHOLD
        else:
            threshold = 1000000
            

        if pink_pixel_count > threshold:
            state |= self.state.PINK
            if (self.state.last_state & self.state.PINK) == 0:
                state |= self.state.PINK_ON

        state |= self.state.DRIVING

        self.state.last_state = self.state.current_state
        self.state.current_state = state

        return cv_image

    def driving(self, cv_image):
        """Function for the DRIVING state"""
        contour_colour = (0, 255, 0)
        lost_left = False

        mask = cv2.inRange(cv_image, LOWER_ROAD, UPPER_ROAD)

        mask = mask[IMAGE_HEIGHT - CROP_AMOUNT : IMAGE_HEIGHT - CROP_AMOUNT + 100, :]

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
                    return
                centroid_x = int(M["m10"] / M["m00"])
                y_offset = IMAGE_HEIGHT - CROP_AMOUNT
                largest_contour[:, 0, 1] += y_offset
                # draw contour onto image
                # cv2.drawContours(cv_image, [largest_contour], -1, contour_colour, 2)
                # calculate the error
                error = (
                    centroid_x - IMAGE_WIDTH / 2
                )  # positive means the car should turn right

        else:  # no contours detected, so set error directly
            error = -IMAGE_WIDTH / 2 if self.previous_error < 0 else IMAGE_WIDTH / 2

        # TODO: smooth this out, so when it's lost it doesnt jerk
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

        # cv2.imshow("Driving Image", cv_image)
        # cv2.waitKey(3)

        # PID controller
        move = Twist()

        derivative = error - self.previous_error
        self.previous_error = error

        move.angular.z = -(KP * error + KD * derivative)

        # decrease linear speed as angular speed increases from a max of 3 down to 1.xx? if abs(error) > 400
        move.linear.x = max(0, MAX_SPEED - SPEED_DROP * abs(error))

        self.move_pub.publish(move)

    def offroad_driving(self, cv_image, speed_drop=SPEED_DROP_OFFROAD, y_mult_cutoff=Y_MULT_CUTOFF, mountain=False, max_speed=MAX_SPEED):
        # Check whether clue has been submitted
        if self.state.clue_in_view and not self.state.offroad_clue_found:
            if self.state.Location.OFFROAD == self.state.current_location:
                print("LOOKING FOR CLUE")
                # Stop driving, and turn to face the clue
                move = Twist()
                move.angular.z = 2.5
                self.move_pub.publish(move)
            elif self.state.Location.MOUNTAIN == self.state.current_location:
                # Drive directly towards the clue, by moving towards it's centroid
                print("DRIVING TOWARDS CLUE")

                # Find the centroid of the clue
                centroid = self.clue_centroid
                # Find the error
                error = centroid[0] - IMAGE_WIDTH / 2
                # PID controller

                move = Twist()
                move.angular.z = -error * 0.002
                move.linear.x = 0.9
                self.move_pub.publish(move)

            return

        total_error = 0

        lines = self.process_image(cv_image)

        if lines is not None:
            unclassified_lines = set()
            for line in lines:
                unclassified_lines.add(tuple(line[0]))

            left_lines, right_lines = self.classify_sets(unclassified_lines)

            for lines in left_lines:
                x1, y1, x2, y2 = lines
                cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for lines in right_lines:
                x1, y1, x2, y2 = lines
                cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # cv2.imshow("Desert Image", cv_image)
            # cv2.waitKey(3)

            # calculate the error by representing each side as a single line
            # Then, find the "moment" of the lines,
            # and use that as the error
            # Lines should have a higher weighting when they are lower
            # and a lower weighting when they are higher
            # Lines will have a higher weighting the closer they are to the center
            # The left and rights lines will have different "neutral"
            # angles, due to perspective

            neutral_angle = 30  # degrees
            if mountain:
                neutral_angle = 38
            neutral_x = IMAGE_WIDTH // 4
            angle_exp = 0.45
            y_mult_intercept = 0.8
            y_mult_exp = 0.97

            right_error = 0
            left_error = 0

            # Find the equivalent line for the left lines by finding the
            # bottom and top point
            if len(left_lines) > 0:
                y_max = 0
                y_min = IMAGE_HEIGHT
                bottom_point = None
                top_point = None
                for line in left_lines:
                    if line[1] > y_max:
                        y_max = line[1]
                        bottom_point = line[0:2]
                    if line[3] > y_max:
                        y_max = line[3]
                        bottom_point = line[2:4]
                    if line[1] < y_min:
                        y_min = line[1]
                        top_point = line[0:2]
                    if line[3] < y_min:
                        y_min = line[3]
                        top_point = line[2:4]

                # Draw the bottom and top points onto image
                cv2.circle(
                    cv_image, (bottom_point[0], bottom_point[1]), 5, (0, 0, 255), -1
                )

                # Find the center between the two points, aand the slope
                left_x = (bottom_point[0] + top_point[0]) // 2
                left_y = (bottom_point[1] + top_point[1]) // 2
                if top_point[0] == bottom_point[0]:
                    left_slope = 1000000
                else:
                    left_slope = (top_point[1] - bottom_point[1]) / (
                        top_point[0] - bottom_point[0]
                    )

                # If the length of the combined line is too short, calculate the angle by 
                # taking the average of the rest of the lines
                if self.line_length((bottom_point[0], bottom_point[1], top_point[0], top_point[1])) < 300: 
                    # last_slope = left_slope
                    left_slope = 0
                    for line in left_lines:
                        left_slope += (line[3] - line[1]) / (line[2] - line[0])
                    left_slope /= len(left_lines)

                left_angle = np.arctan(left_slope) * 180 / np.pi  # This will output
                if left_angle < 0:
                    left_angle += 180

                left_angle = 180 - left_angle

                # ERROR IS POSITIVE IF THE CAR NEEDS TO TURN RIGHT

                lateral_error = left_x - neutral_x
                angle_error = (neutral_angle - left_angle) * abs(
                    neutral_angle - left_angle
                ) ** angle_exp

                if mountain:
                    if abs(angle_error) > 200:
                        angle_error = 200 * angle_error / abs(angle_error)


                # angle_error should be bigger the lower the line is
                y_mult = (
                    max(
                        (left_y - y_mult_cutoff)
                        * abs((left_y - y_mult_cutoff)) ** y_mult_exp
                        / IMAGE_HEIGHT,
                        0,
                    )
                    + y_mult_intercept
                )
                # multiply by a confidence factor depending on number of lines
                # This will be around 0.25 for 1 line, then 0.5 for 2, and 1 for more
                confidence_factor = (len(left_lines) / (
                    len(left_lines) + len(right_lines)
                )) ** 1.8 * 1.5

                left_error = (
                    OKX * lateral_error + OKY * y_mult * angle_error
                ) * confidence_factor

                cv2.putText(
                    cv_image,
                    "lateral error: %d . angle error: %.1f . y_mult: %.1f"
                    % (lateral_error, angle_error, y_mult),
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Find the equivalent line for the right lines by finding the
            # bottom and top point
            if len(right_lines) > 0:
                y_max = 0
                y_min = IMAGE_HEIGHT
                bottom_point = None
                top_point = None
                for line in right_lines:
                    if line[1] > y_max:
                        y_max = line[1]
                        bottom_point = line[0:2]
                    if line[3] > y_max:
                        y_max = line[3]
                        bottom_point = line[2:4]
                    if line[1] < y_min:
                        y_min = line[1]
                        top_point = line[0:2]
                    if line[3] < y_min:
                        y_min = line[3]
                        top_point = line[2:4]

                cv2.circle(
                    cv_image, (bottom_point[0], bottom_point[1]), 5, (255, 0, 0), -1
                )
                # Find the center between the two points, and the slope
                right_x = (bottom_point[0] + top_point[0]) // 2
                right_y = (bottom_point[1] + top_point[1]) // 2
                if top_point[0] == bottom_point[0]:
                    right_slope = 1000000
                else:
                    right_slope = (top_point[1] - bottom_point[1]) / (
                        top_point[0] - bottom_point[0]
                    )

                # If the length of the combined line is too short, calculate the angle by 
                # taking the average of the rest of the lines
                if self.line_length((bottom_point[0], bottom_point[1], top_point[0], top_point[1])) < 300: 
                    # last_slope = right_slope
                    right_slope = 0
                    for line in right_lines:
                        right_slope += (line[3] - line[1]) / (line[2] - line[0])
                    right_slope /= len(right_lines)

                right_angle = np.arctan(right_slope) * 180 / np.pi  # This will output
                if right_angle < 0:
                    right_angle += 180
                right_angle = 180 - right_angle

                # ERROR IS POSITIVE IF THE CAR NEEDS TO TURN RIGHT

                lateral_error = right_x - (IMAGE_WIDTH - neutral_x)
                angle_error = ((180 - neutral_angle) - right_angle) * abs(
                    (180 - neutral_angle) - right_angle
                ) ** angle_exp
                # angle_error should be bigger the lower the line is
                y_mult = (
                    max(
                        (right_y - y_mult_cutoff)
                        * abs((right_y - y_mult_cutoff)) ** y_mult_exp
                        / IMAGE_HEIGHT,
                        0,
                    )
                    + y_mult_intercept
                )
                confidence_factor = len(right_lines) / (
                    len(left_lines) + len(right_lines)
                )

                if mountain:
                    if y_mult > 30:
                        y_mult = 30

                right_error = (
                    OKX * lateral_error + OKY * y_mult * angle_error
                ) * confidence_factor

                # write words onto image at bottom right
                cv2.putText(
                    cv_image,
                    "lateral error: %d . angle error: %.1f . y_mult: %.1f"
                    % (lateral_error, angle_error, y_mult),
                    (IMAGE_WIDTH - 900, IMAGE_HEIGHT - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            total_error = left_error + right_error
            if mountain:
                total_error = left_error*0.45 + right_error*0.55 # mountain error is weighted more by the right line
                if total_error > 100:
                    total_error = 80

            cv2.imshow("Offroad Image", cv_image)
            cv2.waitKey(3)
            # Draw error onto image
            cv2.putText(
                cv_image,
                "Total error: %d" % total_error,
                (10, IMAGE_HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        move = Twist()

        total_error = -total_error  # correct for the way twist works
        derivative = total_error - self.previous_error
        self.previous_error = total_error

        move.angular.z = OKP * KP * total_error + KD * OKD * derivative
        direction = "right" if move.angular.z < 0 else "left"
        move.linear.x = max(0, max_speed - speed_drop * abs(total_error))

        cv2.putText(
            cv_image,
            "Angular velocity: %.2f, %s, Linear velocity: %.2f"
            % (move.angular.z, direction, move.linear.x),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # cv2.imshow("Offroad Image", cv_image)
        # cv2.waitKey(3)
        # # Save the image
        # cv2.imwrite(
        #     CAPTURE_PATH + "offroad_images/offroad%d.png" % self.img_count, cv_image
        # )
        # self.img_count += 1

        self.move_pub.publish(move)

    def classify_sets(self, unclassified_lines):
        left_lines = set()
        left_lines_2 = set()
        right_lines = set()
        right_lines_2 = set()

        rejection_angle = np.pi / 2 - 0.3

        # classify the first and second left lines
        left_line = None
        left_line_2 = None
        for line in unclassified_lines:
            if self.line_angle(line) > (np.pi - rejection_angle):
                if line[2] < IMAGE_WIDTH // 2 and line[0] < IMAGE_WIDTH // 2:
                    if left_line is None:
                        left_line = line
                    elif self.line_length(line) > self.line_length(left_line):
                        if left_line_2 is None and not self.lines_same(left_line, line):
                            left_line_2 = left_line
                        left_line = line
                    # If the line is a near duplicate, don't add it
                    elif left_line_2 is None and not self.lines_same(left_line, line):
                        left_line_2 = line
                    # If the line is a near duplicate, don't add it
                    elif (
                        left_line_2 is not None
                        and self.line_length(line) > self.line_length(left_line_2)
                        and not self.lines_same(left_line, line)
                    ):
                        left_line_2 = line

        if left_line is not None:
            left_lines.add(left_line)

        if left_line_2 is not None:
            left_lines_2.add(left_line_2)

        # classify the first and second right lines
        right_line = None
        right_line_2 = None
        for line in unclassified_lines:
            if self.line_angle(line) < rejection_angle:
                if line[2] > IMAGE_WIDTH // 2 and line[0] > IMAGE_WIDTH // 2:
                    if right_line is None:
                        right_line = line
                    elif self.line_length(line) > self.line_length(right_line):
                        if right_line_2 is None and not self.lines_same(
                            right_line, line
                        ):
                            right_line_2 = right_line
                        right_line = line
                    elif right_line_2 is None and not self.lines_same(right_line, line):
                        right_line_2 = line
                    elif (
                        right_line_2 is not None
                        and self.line_length(line) > self.line_length(right_line_2)
                        and not self.lines_same(right_line, line)
                    ):
                        right_line_2 = line

        if right_line is not None:
            right_lines.add(right_line)

        if right_line_2 is not None:
            right_lines_2.add(right_line_2)

        # Check between left and right line which is lower - classify that set first
        left_y_max = 0
        right_y_max = 0

        for line in left_lines:
            if line[1] > left_y_max:
                left_y_max = line[1]
            if line[3] > left_y_max:
                left_y_max = line[3]

        for line in right_lines:
            if line[1] > right_y_max:
                right_y_max = line[1]
            if line[3] > right_y_max:
                right_y_max = line[3]

        # Determine the order of operations

        if left_y_max > right_y_max:
            first_set = left_lines
            second_set = right_lines
            first_set_2 = left_lines_2
            second_set_2 = right_lines_2
            unclassified_lines -= left_lines

        else:
            first_set = right_lines
            second_set = left_lines
            first_set_2 = right_lines_2
            second_set_2 = left_lines_2
            unclassified_lines -= right_lines

        # Classify first set (first_set)
        self.add_lines_to_set(unclassified_lines, first_set)

        # Check if unclassified_lines contains first_set_2
        # If so, classify first_set_2
        if unclassified_lines & first_set_2:
            self.add_lines_to_set(unclassified_lines, first_set_2)

        # If first_set_2 is larger than first_set, use it instead
        if len(first_set_2) > len(first_set):
            # Remove all lines from first_set and replace with first_set_2
            first_set.clear()
            first_set.update(first_set_2)

        # If second_set is not in unclassified_lines, it is already classified
        # so we can't use it
        if unclassified_lines & second_set:
            # Repeat for second_set
            self.add_lines_to_set(unclassified_lines, second_set)
        else:
            second_set -= (
                second_set  # Make it empty while keeping the link to the original set
            )

        # Check if unclassified_lines contains line_2 (second_set_2)
        # If so, classify second_set_2
        if unclassified_lines & second_set_2:
            self.add_lines_to_set(unclassified_lines, second_set_2)
        else:
            second_set_2 -= second_set_2

        # Replace second_set with second_set_2 if it is larger
        if len(second_set_2) > len(second_set):
            second_set.clear()
            second_set.update(second_set_2)

        # If both sets are empty, check remaining unclassified lines
        if len(second_set) == 0 and len(second_set_2) == 0:
            second_set -= second_set
            new_line = None
            for line in unclassified_lines:
                if left_y_max > right_y_max:  # unclassified lines are right
                    if self.line_angle(line) < rejection_angle and line[2] > IMAGE_WIDTH // 2 and line[0] > IMAGE_WIDTH // 2:
                        if new_line is None:
                            new_line = line
                        elif self.line_length(line) > self.line_length(new_line):
                            new_line = line
                else:  # unclassified lines are left
                    if self.line_angle(line) > np.pi - rejection_angle and line[2] < IMAGE_WIDTH // 2 and line[0] < IMAGE_WIDTH // 2:
                        if new_line is None:
                            new_line = line
                        elif self.line_length(line) > self.line_length(new_line):
                            new_line = line

            if new_line is not None:
                second_set.add(new_line)

                self.add_lines_to_set(unclassified_lines, second_set)

        return left_lines, right_lines

    def add_lines_to_set(self, unclassified, set):
        line_classified = True
        while line_classified:
            line_classified = False
            for line in list(unclassified):
                # Check if the line is in the region of a left line
                is_near = self.near_set(line, set)
                if is_near:
                    set.add(line)
                    unclassified.remove(line)
                    line_classified = True

    def line_length(self, line):
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def line_angle(self, line):
        """Returns the angle of the line in radians
        If angled to the right, the angle will be from 0 to pi/2
        If angled to the left, the angle will be from pi/2 to pi"""
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1)  # from -pi to pi
        if angle < 0:
            angle += np.pi
        return angle

    def near_set(self, line, set):
        """Returns whether the line is in the region of a line in the set"""
        if len(set) == 0:
            return False

        for set_line in set:
            # If either point of the line is within set_threshold pixels of any point on
            # the set line, return true
            if self.point_near_line(line[0:2], set_line) or self.point_near_line(
                line[2:4], set_line
            ):
                return True

        return False

    def point_near_line(self, point, line):
        x1, y1, x2, y2 = line
        x, y = point

        set_threshold = 35

        # If the point is too far outside the points of the line, return false
        if (x + set_threshold < x1 and x + set_threshold < x2) or (
            x - set_threshold > x1 and x - set_threshold > x2
        ):
            return False
        elif (y + set_threshold < y1 and y + set_threshold < y2) or (
            y - set_threshold > y1 and y - set_threshold > y2
        ):
            return False
        # Otherwise, calculate distance to line, and return if it is small enough
        else:
            distance = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / np.sqrt(
                (y2 - y1) ** 2 + (x2 - x1) ** 2
            )
            return distance < set_threshold

    def crop_to_floor(self, image):
        # Mask for floor
        floor_mask = cv2.inRange(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV), FLOOR_LOWER_HSV, FLOOR_UPPER_HSV
        )
        # clean up the mask
        floor_mask = cv2.erode(floor_mask, KERNEL_2, iterations=1)

        # Mask for green
        green_mask = cv2.inRange(image, GREEN_LOWER, GREEN_UPPER)
        # clean up the mask
        green_mask = cv2.dilate(green_mask, KERNEL_5, iterations=4)

        # set all images outside the mask to black
        total_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(green_mask))

        # Extra safety mask with manual values
        if self.state.extra_mask_enabled:
            extra_mask = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)*255
            extra_mask[IMAGE_HEIGHT//2 - 100 : IMAGE_HEIGHT//2 + 100, :300] = 0
            total_mask = cv2.bitwise_and(total_mask, extra_mask)

        image = cv2.bitwise_and(image, image, mask=total_mask)

        # Find the y value of the lowest point of the mask
        _, min_y, _, _ = cv2.boundingRect(total_mask)

        return image[min_y:, :], total_mask[min_y:, :], min_y

    def process_image(self, image):
        image, mask, min_y = self.crop_to_floor(image)

        # find canny edges for the image and the mask

        blurred = cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=55)
        blurred = cv2.bilateralFilter(blurred, d=7, sigmaColor=75, sigmaSpace=55)

        edges = cv2.Canny(blurred, 50, 150)

        edges_mask = cv2.Canny(mask, 50, 150)

        # dilate the edges mask
        edges_mask = cv2.dilate(edges_mask, KERNEL_3, iterations=2)

        # remove edeges from edges that are also in edges_mask
        edges[edges_mask == 255] = 0

        # Use Hough Line Transform to find line segments on the edges
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=20
        )

        # Draw lines on a grayscale copy of the original image
        image_with_lines = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_with_lines, (x1, y1), (x2, y2), 255, 2)

        # set all pixels that are not white to black
        image_with_lines[image_with_lines < 255] = 0

        # Use Hough Line Transform to find line segments on the image with lines
        lines = cv2.HoughLinesP(
            image_with_lines,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=60,
            maxLineGap=20,
        )

        if lines is None:
            return None

        # Add min_y to the y values of the lines to account for the crop
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line[0][1] += min_y
            line[0][3] += min_y

        return lines

    def desert(self, cv_image):
        if self.state.current_time - self.state.last_pink_time < 0.5:
            self.offroad_driving(cv_image)
            return

        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([110, 3, 3])
        image_split = 2
        pink_thresh = 1
        pink_thresh2 = 20000
        time_thresh = 0.6
        lower_pink = np.array([200, 0, 200])
        upper_pink = np.array([255, 150, 255])
        lower_tunnel = np.array([80, 100, 185])
        upper_tunnel = np.array([95, 120, 195])
        statey = "blue"
        set_it_up = False

        blue_mask = cv2.inRange(cv_image, lower_blue, upper_blue)
        pink_mask = cv2.inRange(cv_image, lower_pink, upper_pink)
        tunnel_mask = cv2.inRange(cv_image, lower_tunnel, upper_tunnel)
        tunnel_mask = cv2.dilate(tunnel_mask, np.ones((3, 3), np.uint8), iterations=5)
        pink_pixel_count = cv2.countNonZero(pink_mask)

        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if (
            pink_pixel_count > pink_thresh
            and self.state.current_time - self.state.last_pink_time > time_thresh
        ):
            statey = "pink"
            contours, _ = cv2.findContours(
                pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        if (
            pink_pixel_count > pink_thresh2 or self.tunnel_time
        ) and self.state.current_time - self.state.last_pink_time > time_thresh:
            self.tunnel_time = True
            if self.pink_tunnel_reached == 0:
                self.pink_tunnel_reached = self.state.current_time
            set_it_up = True
            statey = "tunnel"
            contours, _ = cv2.findContours(
                tunnel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        if len(contours) > 0:
            if self.tunnel_time:
                x_coordinates = []
                for contour in contours:
                    # Calculate the moments of the contour
                    M = cv2.moments(contour)

                    # Calculate centroid coordinates
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        x_coordinates.append(cx)

                # Calculate the average x-coordinate
                if x_coordinates:
                    centroid_x = np.mean(x_coordinates)

            else:
                max_contour = max(contours, key=cv2.contourArea)
                # cv2.drawContours(cv_image, [max_contour], -1, (0, 0, 255), 2)
                M = cv2.moments(max_contour)
                if M["m00"] == 0:
                    return
                centroid_x = int(M["m10"] / M["m00"])

            cv2.circle(
                cv_image,
                (int(IMAGE_WIDTH / image_split), IMAGE_HEIGHT // 2),
                5,
                (0, 255, 0),
                -1,
            )
            cv2.circle(
                cv_image, (int(centroid_x), IMAGE_HEIGHT // 2), 5, (0, 0, 255), -1
            )
            cv2.putText(
                cv_image,
                str(pink_pixel_count),
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                cv_image,
                statey,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            error = IMAGE_WIDTH / image_split - centroid_x
        else:
            error = IMAGE_WIDTH / image_split

        move = Twist()
        derivative = error - self.previous_error
        self.previous_error = error

        # positive means the car should turn right
        move.angular.z = KP * error + KD * derivative

        move.linear.x = max(0, 1.1 - SPEED_DROP * abs(error))

        if statey == "pink":
            move.linear.x = 0.5

        if set_it_up:
            if self.state.current_time - self.pink_tunnel_reached > 0.8:
                if error > -2:
                    move.linear.x = 0.0
                    move.angular.z = 0.2 * KP * error + KD * derivative
                else:
                    # Change state to mountain
                    self.state.current_location = self.state.Location.MOUNTAIN
                    self.state.offroad_clue_found = False
                    # print("SWITCHED TO MOUNTAIN STATE")
                    move.linear.x = 0.5
                    move.angular.z = 0.0
                    self.move_pub.publish(move)
                    rospy.sleep(1)
            else:
                move.linear.x = 0.3
                move.angular.z = 0.0

        self.move_pub.publish(move)

    def mountain(self, cv_image):
        if not self.state.tunnel_state:
            self.mountain_driving(cv_image, SPEED_DROP_MOUNTAIN, Y_MULT_CUTOFF_MOUNTAIN, True, 0.6)
            return
        
        if self.state.first_mountain_call:
            move = Twist()
            move.linear.x = 0.6
            self.move_pub.publish(move)
            print("STRAIGHT TO TUNNEL", self.state.current_time)
            rospy.sleep(0.3)
            self.state.first_mountain_call = False
            print("STARTING TUNNEL STATE", self.state.current_time)
            return

        lower_sky = np.array([180, 120, 100])
        upper_sky = np.array([210, 150, 140])

        sky_mask = cv2.inRange(cv_image, lower_sky, upper_sky)
        sky_mask = cv2.dilate(sky_mask, np.ones((3, 3), np.uint8), iterations=3)

        contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        error = 0

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 23000:
                self.state.tunnel_state = False
                print("EXITED TUNNEL STATE")
                return
            
            else:
                M = cv2.moments(largest_contour)
                if M["m00"] == 0:
                    return
                centroid_x = int(M["m10"] / M["m00"])
                # draw contour onto image
                cv2.drawContours(cv_image, [largest_contour], -1, (0, 0, 255), 2)
                # calculate the error
                error = (
                    centroid_x - IMAGE_WIDTH / 2
                )  # positive means the car should turn right

            error = centroid_x - IMAGE_WIDTH // 2

    
        total_error = (-error)*0.01 + 0.1 # Offset to the left by a bit
        derivative = total_error - self.previous_error_t
        self.previous_error_t = total_error

        move = Twist()
        move.angular.z = total_error + derivative * 0.5
        move.linear.x = 0.3
        self.move_pub.publish(move)

        # print("Error: %.1f . centroid: %.1f .contour area: %.1f ." % (total_error, centroid_x, cv2.contourArea(largest_contour)))

        # cv2.putText(
        #             cv_image,
        #             "Error: %.1f . centroid: %.1f . contour area: %.1f " % (total_error, centroid_x, cv2.contourArea(largest_contour)),
        #             (IMAGE_WIDTH - 900, IMAGE_HEIGHT - 50),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (0, 0, 255),
        #             2,
        #             cv2.LINE_AA,
        #         )
        
        # cv2.imwrite(CAPTURE_PATH + "mountain_images/tunnel_img%d.png" % self.img_count, cv_image)
        # self.img_count += 1


        # cv2.imshow("Tunnel", cv_image)
        # cv2.waitKey(3)
            
    def lines_same(self, line1, line2):
        # If both points are within 50 pixels of each other, return true
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        distance1 = min(
            np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2),
            np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2),
        )
        distance2 = min(
            np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
            np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2),
        )
        return distance1 < 50 and distance2 < 50

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

    def update_clue(self, cv_image):
        """Crops the image to the clue, if one is found
        Stores the best clue in state"""

        # Convert BGR to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        uh = 130
        us = 255
        uv = 255
        lh = 120
        ls = 100
        lv = 70
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # find contours in the mask
        cnts, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        # If on the offroad section, and no clue has been found,
        # Look for any blue contours, and if there are any big enough,
        # assume it is a clue
        if (
            self.state.current_location == self.state.Location.OFFROAD
            and self.state.offroad_clue_found is False
            and self.state.current_time - self.state.start_offroad_time > 5
        ):
            for i in range(len(cnts)):
                area = cv2.contourArea(cnts[i])
                # check if the contour has children
                if self.state.Location.OFFROAD == self.state.current_location:
                    if hierarchy[0][i][2] != -1:
                        continue

                if area > 4000:
                    self.state.clue_in_view = True
                    if self.state.clue_first_seen:
                        self.state.clue_first_seen = False
                        move = Twist()
                        move.linear.x = 0.4
                        move.angular.z = 0.3
                        self.move_pub.publish(move)
                        self.state.turn_flag = True
                        rospy.sleep(0.5)
                        self.state.turn_flag = False
                    imgCopy = cv_image.copy()
                    # cv2.drawContours(imgCopy, [cnts[i]], -1, (0, 0, 255), 4)
                    # cv2.imshow("Clue", imgCopy)
                    # cv2.waitKey(3)
                    break
        # elif (
        #     self.state.current_location == self.state.Location.MOUNTAIN
        #     and self.state.offroad_clue_found is False
        #     and self.state.current_time - self.state.start_offroad_time > 5
        # ):
        #     print("searching for clue")
        #     # Get the largest contour
        #     largest_contour = max(cnts, key=cv2.contourArea)
        #     area = cv2.contourArea(largest_contour)
        #     if area > 4000:
        #         print("clue found") 
        #         # Find the centroid of the contour
        #         M = cv2.moments(largest_contour)
        #         if M["m00"] == 0:
        #             return
        #         self.clue_centroid = (
        #             int(M["m10"] / M["m00"]),
        #             int(M["m01"] / M["m00"]),
        #         )
        #         self.state.clue_in_view = True
            
                

        # Find the contours that are holes (i.e. have a parent)
        holes = []
        for i in range(len(cnts)):
            if hierarchy[0][i][3] >= 0:
                holes.append(cnts[i])

        # If something might be a clue, find the largest contour
        if len(holes) > 0:
            clue_border = max(holes, key=cv2.contourArea)
            clue_size = cv2.contourArea(clue_border)
            # Check area is big enough (to avoid false positives, or clues that are too far)
            if clue_size < 10000:
                return

            if self.state.Location.OFFROAD == self.state.current_location and not self.state.turn_flag:
                self.state.offroad_clue_found = True

            # THE FOLLOWING HAPPENS IF A CLUE IS FOUND
            # If the area is bigger than the previous biggest area, update the best clue
            # and reset the time since the clue improved
            if clue_size > self.state.max_area:
                self.state.max_area = clue_size
                self.state.clue_improved_time = self.state.current_time

                # Template that the clue will be warped onto
                clue = np.zeros((400, 600, 3), np.uint8)

                top_right, bottom_right, bottom_left, top_left = self.find_clue_corners(
                    clue_border
                )

                input_pts = np.float32([top_right, bottom_right, bottom_left, top_left])
                output_pts = np.float32(
                    [
                        [clue.shape[1], 0],
                        [clue.shape[1], clue.shape[0]],
                        [0, clue.shape[0]],
                        [0, 0],
                    ]
                )

                matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
                self.state.best_clue = cv2.warpPerspective(
                    cv_image,
                    matrix,
                    (clue.shape[1], clue.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )

        return

    def submit_clue(self):
        # cv2.imshow("Best Clue", self.state.best_clue)
        # Get the words from the clue
        type, clue = self.find_words()

        print("Clue: %s" % clue)

        # Find clue number using the type dictionary
        # In case the type is not found exactly, find the closest match
        if type in CLUE_TYPES:
            type_num = CLUE_TYPES[type]
        else:
            type_num = 0
            min_dist = 100
            for key in CLUE_TYPES:
                dist = Levenshtein.distance(key, type)
                if dist < min_dist:
                    min_dist = dist
                    type_num = CLUE_TYPES[key]

        # Publish the clue
        self.score_pub.publish("%s,%s,%d,%s" % (TEAM_NAME, PASSWORD, type_num, clue))

        self.state.clues_submitted.append(type_num)

        if type_num == 4:
            self.state.detection_enabled = False
        elif type_num == 6:
            self.state.extra_mask_enabled = False
        elif type_num == 8:
            self.state.finished = True

    def find_words(self):
        clue = self.state.best_clue

        clue_plate = clue[200:395, 5:595].copy()
        type_plate = clue[5:200, 5:595].copy()

        clue_hsv = cv2.cvtColor(clue_plate, cv2.COLOR_BGR2HSV)
        type_hsv = cv2.cvtColor(type_plate, cv2.COLOR_BGR2HSV)

        uh = 130
        us = 255
        uv = 255
        lh = 110
        ls = 100
        lv = 80

        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        type_mask = cv2.inRange(type_hsv, lower_hsv, upper_hsv)
        clue_mask = cv2.inRange(clue_hsv, lower_hsv, upper_hsv)

        # Dilate then erode to fill them in (?)
        for i in range(10):
            type_mask = cv2.dilate(type_mask, KERNEL_3, iterations=1)
            clue_mask = cv2.dilate(clue_mask, KERNEL_3, iterations=1)

            type_mask = cv2.erode(type_mask, KERNEL_3, iterations=1)
            clue_mask = cv2.erode(clue_mask, KERNEL_3, iterations=1)

        # Find contours (letters)

        type_cnts, _ = cv2.findContours(
            type_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        clue_cnts, _ = cv2.findContours(
            clue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Remove contours that are too small

        type_cnts = [c for c in type_cnts if cv2.contourArea(c) > 100]
        clue_cnts = [c for c in clue_cnts if cv2.contourArea(c) > 100]

        # The bounding boxes of the contours (a box with a letter in it)
        clueBoundingBoxes = [cv2.boundingRect(c) for c in clue_cnts]
        typeBoundingBoxes = [cv2.boundingRect(c) for c in type_cnts]

        # sort the contours from left-to-right
        typeBoundingBoxes = sorted(
            (typeBoundingBoxes), key=lambda b: b[0], reverse=False
        )
        clueBoundingBoxes = sorted(
            (clueBoundingBoxes), key=lambda b: b[0], reverse=False
        )

        # If any bounding boxes are too big, split it (in case 2 letters' contours are connected)
        for i in range(len(clueBoundingBoxes)):
            if clueBoundingBoxes[i][2] > 60:
                box1 = (
                    clueBoundingBoxes[i][0],
                    clueBoundingBoxes[i][1],
                    clueBoundingBoxes[i][2] // 2,
                    clueBoundingBoxes[i][3],
                )
                box2 = (
                    clueBoundingBoxes[i][0] + clueBoundingBoxes[i][2] // 2,
                    clueBoundingBoxes[i][1],
                    clueBoundingBoxes[i][2] // 2,
                    clueBoundingBoxes[i][3],
                )
                clueBoundingBoxes.remove(clueBoundingBoxes[i])
                clueBoundingBoxes.insert(i, box1)
                clueBoundingBoxes.insert(i + 1, box2)

        for i in range(len(typeBoundingBoxes)):
            if typeBoundingBoxes[i][2] > 60:
                box1 = (
                    typeBoundingBoxes[i][0],
                    typeBoundingBoxes[i][1],
                    typeBoundingBoxes[i][2] // 2,
                    typeBoundingBoxes[i][3],
                )
                box2 = (
                    typeBoundingBoxes[i][0] + typeBoundingBoxes[i][2] // 2,
                    typeBoundingBoxes[i][1],
                    typeBoundingBoxes[i][2] // 2,
                    typeBoundingBoxes[i][3],
                )
                typeBoundingBoxes.remove(typeBoundingBoxes[i])
                typeBoundingBoxes.insert(i, box1)
                typeBoundingBoxes.insert(i + 1, box2)

        # Letter Template
        letter = np.zeros((130, 80, 3), np.uint8)

        letter_imgs = []

        # For each rectangle (which is a letter), warp perspective to get the letter into the template
        for i in range(len(typeBoundingBoxes)):
            (x, y, w, h) = typeBoundingBoxes[i]
            input_pts = np.float32([[x + w, y], [x + w, y + h], [x, y + h], [x, y]])
            output_pts = np.float32(
                [
                    [letter.shape[1], 0],
                    [letter.shape[1], letter.shape[0]],
                    [0, letter.shape[0]],
                    [0, 0],
                ]
            )
            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
            letter = cv2.warpPerspective(
                type_plate,
                matrix,
                (letter.shape[1], letter.shape[0]),
                flags=cv2.INTER_LINEAR,
            )
            letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
            letter_imgs.append(np.array(letter))

        for i in range(len(clueBoundingBoxes)):
            (x, y, w, h) = clueBoundingBoxes[i]
            input_pts = np.float32([[x + w, y], [x + w, y + h], [x, y + h], [x, y]])
            output_pts = np.float32(
                [
                    [letter.shape[1], 0],
                    [letter.shape[1], letter.shape[0]],
                    [0, letter.shape[0]],
                    [0, 0],
                ]
            )
            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
            letter = cv2.warpPerspective(
                clue_plate,
                matrix,
                (letter.shape[1], letter.shape[0]),
                flags=cv2.INTER_LINEAR,
            )
            letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
            letter_imgs.append(np.array(letter))

        letter_imgs = np.expand_dims(np.array(letter_imgs), axis=-1)

        outputs = []

        self.lite_model.resize_tensor_input(
            self.lite_model.get_input_details()[0]["index"],
            [len(letter_imgs), 130, 80, 1],
        )
        self.lite_model.allocate_tensors()

        self.lite_model.set_tensor(
            self.input_details[0]["index"], letter_imgs.astype(np.float32)
        )
        self.lite_model.invoke()

        outputs = self.lite_model.get_tensor(self.output_details[0]["index"])
        outputs = np.array(outputs)

        predicted_values = np.argmax(outputs, axis=1)
        # turn the predicted values into the characters/numbers
        predicted_values = [
            chr(x + 65) if x < 26 else str(x - 26) for x in predicted_values
        ]
        predicted_values = "".join(predicted_values)

        type = predicted_values[: len(typeBoundingBoxes)]
        clue = predicted_values[len(typeBoundingBoxes) :]

        return type, clue

    # TODO: clean this up, make constants, etc. also prob more testing
    def obstacle_detection(self, cv_image):
        """Function for the OBSTACLE state"""
        frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # apply masks for raod lines and white
        frame = cv2.erode(frame, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.inRange(frame, 70, 115)
        frame[mask > 0] = [82]
        mask = cv2.inRange(frame, 160, 255)
        frame[mask > 0] = [82]

        # cv2.imshow("Frame pre-erode", frame)
        # Erode the image to remove artifacts
        frame = cv2.erode(frame, np.ones((3, 3), np.uint8), iterations=2)

        # Apply background subtraction
        fgmask = self.fgbg.apply(frame)

        # Segment the image
        segmented_image = cv2.bitwise_and(frame, frame, mask=fgmask)

        # crop the segmented image
        segmented_image = segmented_image[
            IMAGE_WIDTH // 2 - 200 : IMAGE_WIDTH // 2 + 200,
            IMAGE_HEIGHT - 300 : IMAGE_HEIGHT,
        ]

        # check how many pixels on in the image
        pixel_count = cv2.countNonZero(segmented_image)

        return pixel_count > OBSTACLE_THRESHOLD and self.state.detection_enabled

    # TODO: Test this and prob do something better then stop? also only works for road rn, gets tripped up on offroad
    def avoid_obstacle(self):
        """Function for avoiding obsatcle"""
        self.move_pub.publish(Twist())

    def find_clue_corners(self, clue_border):
        # Simplify the contour to 4 points
        epsilon = 0.1 * cv2.arcLength(clue_border, True)
        approx = cv2.approxPolyDP(clue_border, epsilon, True)

        while len(approx) != 4:
            if len(approx) > 4:
                epsilon = epsilon * 1.05
                approx = cv2.approxPolyDP(clue_border, epsilon, True)
            elif len(approx) < 4:
                epsilon = epsilon * 0.96
                approx = cv2.approxPolyDP(clue_border, epsilon, True)

        # Classify the points as top left, top right, bottom right, bottom left

        # Make a set of the points in the contour
        approx_set = set()
        for i in range(len(approx)):
            approx_set.add(tuple(approx[i][0]))

        # Top left is the point with the smallest sum of x and y:
        top_left = [5000, 5000]
        for point in approx_set:
            if point[0] + point[1] < top_left[0] + top_left[1]:
                top_left = point

        approx_set.remove(top_left)

        # Bottom right is the point with the largest sum of x and y
        bottom_right = [0, 0]
        for point in approx_set:
            if point[0] + point[1] > bottom_right[0] + bottom_right[1]:
                bottom_right = point
        approx_set.remove(bottom_right)

        # Top right is the point with largest x of the remaining points
        top_right = [0, 0]
        for point in approx_set:
            if point[0] > top_right[0]:
                top_right = point
        approx_set.remove(top_right)

        # Bottom left is the only point left
        bottom_left = approx_set.pop()

        return top_right, bottom_right, bottom_left, top_left
    
    def mountain_driving(self, cv_image, speed_drop=SPEED_DROP_OFFROAD, y_mult_cutoff=Y_MULT_CUTOFF, mountain=False, max_speed=MAX_SPEED):

        # If the car is on the top of the mountain (the top half of the image is sky), drive straight 
        # for a bit, then turn left
        # Mask the image using FLOOR_HSV values
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        floor_mask = cv2.inRange(hsv, FLOOR_LOWER_HSV, FLOOR_UPPER_HSV)
        # Find the highest point in the floor mask, by using bounding boxes
        _, min_y, _, _ = cv2.boundingRect(floor_mask)
        if min_y > IMAGE_HEIGHT//2 and self.state.never_on_top:
            # PRINT
            print("ON TOP OF MOUNTAIN")
            # Drive straight for a bit
            move = Twist()
            move.linear.x = 0.5
            self.move_pub.publish(move)
            rospy.sleep(0.8)
            self.state.never_on_top = False
        
        # if self.state.clue_in_view:
        #     # Drive directly towards the clue, by moving towards it's centroid
        #     print("DRIVING TOWARDS CLUE")

        #     # Find the centroid of the clue
        #     centroid = self.clue_centroid
        #     # Find the error
        #     error = centroid[0] - IMAGE_WIDTH / 2
        #     # PID controller

        #     move = Twist()
        #     move.angular.z = -error * 0.002
        #     move.linear.x = 0.9
        #     self.move_pub.publish(move)

        #     return
        
        total_error = 0

        lines = self.process_image(cv_image)

        if lines is not None:
            unclassified_lines = set()
            for line in lines:
                unclassified_lines.add(tuple(line[0]))
                
            left_lines, right_lines = self.classify_sets(unclassified_lines)

            for lines in left_lines:
                x1, y1, x2, y2 = lines
                cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for lines in right_lines:
                x1, y1, x2, y2 = lines
                cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # cv2.imshow("Desert Image", cv_image)
            # cv2.waitKey(3)

            if len(left_lines) == 0:
                lost_left = True

            if len(right_lines) == 0:
                lost_right = True

            for line in left_lines:
                x1, y1, x2, y2 = line
                cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for line in right_lines:
                x1, y1, x2, y2 = line
                cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # calculate the error by representing each side as a single line
            # Then, find the "moment" of the lines,
            # and use that as the error
            # Lines should have a higher weighting when they are lower
            # and a lower weighting when they are higher
            # Lines will have a higher weighting the closer they are to the center
            # The left and rights lines will have different "neutral"
            # angles, due to perspective

            neutral_angle = 45  # degrees
            neutral_x = IMAGE_WIDTH // 4
            y_mult_cutoff = IMAGE_HEIGHT - 350

            right_error = 0
            left_error = 0

            # Find the equivalent line for the left lines by finding the
            # bottom and top point
            if len(left_lines) > 0:
                y_max = 0
                y_min = IMAGE_HEIGHT
                bottom_point = None
                top_point = None
                for line in left_lines:
                    if line[1] > y_max and line[1] < IMAGE_HEIGHT - 150: # Ignore lines that are too close to the bottom
                        y_max = line[1]
                        bottom_point = line[0:2]
                    if line[3] > y_max and line[3] < IMAGE_HEIGHT - 150:
                        y_max = line[3]
                        bottom_point = line[2:4]
                    if line[1] < y_min:
                        y_min = line[1]
                        top_point = line[0:2]
                    if line[3] < y_min:
                        y_min = line[3]
                        top_point = line[2:4]

                if bottom_point is not None and top_point is not None:
                    # Draw the bottom and top points onto image
                    cv2.circle(
                        cv_image, (bottom_point[0], bottom_point[1]), 5, (0, 0, 255), -1
                    )
                # Find the center between the two points, aand the slope
                left_x = (bottom_point[0] + top_point[0]) // 2
                left_y = (bottom_point[1] + top_point[1]) // 2
                if top_point[0] == bottom_point[0]:
                    left_slope = 1000000
                else:
                    left_slope = (top_point[1] - bottom_point[1]) / (
                        top_point[0] - bottom_point[0]
                    )
                left_angle = np.arctan(left_slope) * 180 / np.pi  # This will output
                if left_angle < 0:
                    left_angle += 180

                left_angle = 180 - left_angle

                # ERROR IS POSITIVE IF THE CAR NEEDS TO TURN RIGHT

                lateral_error = left_x - neutral_x
                angle_error = neutral_angle - left_angle
                # angle_error should be bigger the lower the line is
                y_mult = max((left_y - y_mult_cutoff) ** 2 / IMAGE_HEIGHT, 0)
                # if y_mult > 40:
                #     y_mult = 40
                # multiply by a confidence factor depending on number of lines
                # This will be around 0.25 for 1 line, then 0.75 for 2, and 1 for more
                confidence_factor = min(0.25 + 0.7 * len(left_lines), 1)

                # write words onto image at top left
                cv2.putText(
                    cv_image,
                    "lateral error: %d . angle error: %.1f . y_mult: %.1f"
                    % (lateral_error, angle_error, y_mult),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # print("Left angle: ", left_angle)

                # print("Left lateral error: %d . Left angle error: %.1f" % (lateral_error*OKX, angle_error*OKY*y_mult))

                left_error = (
                    lateral_error + 0.2 * y_mult * angle_error
                ) * confidence_factor

            # Find the equivalent line for the right lines by finding the
            # bottom and top point
            if len(right_lines) > 0:
                y_max = 0
                y_min = IMAGE_HEIGHT
                bottom_point = None
                top_point = None
                for line in right_lines:
                    if line[1] > y_max:
                        y_max = line[1]
                        bottom_point = line[0:2]
                    if line[3] > y_max:
                        y_max = line[3]
                        bottom_point = line[2:4]
                    if line[1] < y_min:
                        y_min = line[1]
                        top_point = line[0:2]
                    if line[3] < y_min:
                        y_min = line[3]
                        top_point = line[2:4]

                cv2.circle(
                    cv_image, (bottom_point[0], bottom_point[1]), 5, (255, 0, 0), -1
                )
                # Find the center between the two points, and the slope
                right_x = (bottom_point[0] + top_point[0]) // 2
                right_y = (bottom_point[1] + top_point[1]) // 2
                if top_point[0] == bottom_point[0]:
                    right_slope = 1000000
                else:
                    right_slope = (top_point[1] - bottom_point[1]) / (
                        top_point[0] - bottom_point[0]
                    )
                right_angle = np.arctan(right_slope) * 180 / np.pi  # This will output
                if right_angle < 0:
                    right_angle += 180
                right_angle = 180 - right_angle

                # print("Right angle: ", right_angle)

                # ERROR IS POSITIVE IF THE CAR NEEDS TO TURN RIGHT

                lateral_error = right_x - (IMAGE_WIDTH - neutral_x)
                angle_error = (180 - neutral_angle) - right_angle
                # angle_error should be bigger the lower the line is
                y_mult = max((right_y - y_mult_cutoff) ** 2 / IMAGE_HEIGHT, 0)
                confidence_factor = min(0.25 + 0.5 * len(right_lines), 1)

                right_error = (
                    lateral_error + 0.2 * y_mult * angle_error
                ) * confidence_factor

                # write words onto image at bottom right
                cv2.putText(
                    cv_image,
                    "lateral error: %d . angle error: %.1f . y_mult: %.1f"
                    % (lateral_error, angle_error, y_mult),
                    (IMAGE_WIDTH - 900, IMAGE_HEIGHT - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # print("Right lateral error: %d . Right angle error: %.1f" % (lateral_error*OKX, angle_error*OKY*y_mult))

            total_error = left_error + right_error

        # cv2.imwrite(CAPTURE_PATH + "mountain_images/mount_img%d.png" % self.img_count, cv_image)
        # self.img_count += 1

        move = Twist()

        total_error = -total_error  # correct for the way twist works
        derivative = total_error - self.previous_error
        self.previous_error = total_error

        move.angular.z = 0.4 * 0.017 * total_error + 0.003 * 1 * derivative
        # print("Desert angular speed:", move.angular.z)

        move.linear.x = max(0, 0.4 - 0.00025 * abs(total_error))
        # print("Desert linear speed:", move.linear.x)

        self.move_pub.publish(move)


def main(args):
    rospy.init_node("topic_publisher")
    topic_publisher()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)