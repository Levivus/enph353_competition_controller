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

import Levenshtein

import os
import os.path
from os import path
import inspect

# from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import optimizers

# from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend


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
LOSS_FACTOR = 200
RESPAWN_THRESHOLD = 10
LOWER_PINK = np.array([200, 0, 200], dtype=np.uint8)
UPPER_PINK = np.array([255, 150, 255], dtype=np.uint8)
PINK_THRESHOLD = 100000
LOWER_RED = np.array([0, 0, 200], dtype=np.uint8)
UPPER_RED = np.array([100, 100, 255], dtype=np.uint8)
RED_THRESHOLD = 10000
LOWER_ROAD = np.array([0, 0, 78], dtype=np.uint8)
UPPER_ROAD = np.array([125, 135, 255], dtype=np.uint8)
LOWER_DIRT = np.array([140, 164, 168], dtype=np.uint8)
UPPER_DIRT = np.array([186, 220, 228], dtype=np.uint8)

# PID CONSTANTS
KP = 0.017
KD = 0.003
DKP = 0.4 # desert KP, multiplies KP
MAX_SPEED = 0.3
SPEED_DROP = 0.00055

CLUE_TYPES = {"SIZE": 1, "VICTIM": 2, "CRIME": 3, "TIME": 4, "PLACE": 5, "MOTIVE": 6, "WEAPON": 7, "BANDIT": 8}

PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/"
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

    class Action(Enum):
        DRIVE = 0
        CRY = 1
        RESPAWN = 2
        EXPLODE = 3
        SIT = 4

    # Make state bitfield, so that multiple states can be active at once
    NOTHING = 0b00000000
    DRIVING = 0b00000001
    PINK = 0b00000010
    RED = 0b00000100
    CLUE = 0b00001000
    PINK_ON = 0b00010000

    # location list
    LOCATIONS = [Location.ROAD, Location.OFFROAD, Location.DESERT, Location.MOUNTAIN]

    def __init__(self):
        # Define the initial state
        self.location_count = 0
        self.current_location = self.LOCATIONS[self.location_count]
        self.last_state = self.NOTHING
        self.last_pink_time = 0
        self.best_clue = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), dtype=np.uint8)
        self.clue_improved_time = time.time() + 500
        self.max_area = 0
        self.current_state = self.NOTHING
    
    def set_location(self, location):
        self.current_location = location
        self.location_count = self.LOCATIONS.index(location)

    def choose_action(self):
        if self.current_state & self.NOTHING:
            return self.Action.SIT
        elif self.current_state & self.PINK:
            if (
                self.current_state & self.PINK_ON
                and time.time() - self.last_pink_time > 5
            ):
                self.last_pink_time = time.time()
                self.current_state &= ~self.PINK_ON
                self.location_count += 1
                self.current_location = self.LOCATIONS[self.location_count]
                # elif time.time() - self.last_pink_time > 5:
                #     self.location_count += 1
                #     self.current_location = self.LOCATIONS[self.location_count]
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
        self.count = 0
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
        self.move_pub.publish(Twist())
        # LETTER MODEL
        self.letter_model = tf.keras.models.load_model(ABSOLUTE_PATH + "clue_model.h5", compile=False)
        backend.set_learning_phase(0) # Tell keras it will only be predicting, not training
        self.letter_model.predict(np.zeros((1, 130, 80, 1), dtype=np.uint8)) # Run a prediction to initialize the model

        self.time_start = rospy.wait_for_message("/clock", Clock).clock.secs
        self.score_pub.publish("%s,%s,0,NA" % (TEAM_NAME, PASSWORD))
        self.running = True
        self.spawn_position(HOME)
        print("Done init")

    def clock_callback(self, data):
        """Callback for the clock subscriber
        Publishes the score to the score tracker node when the competition ends
        """
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

        # print("State:", self.state.current_state)
        # print("Action:", action)
        # print("Location:", self.state.current_location)

        if (
            action == State.Action.EXPLODE
        ):  # TODO: shut down the script? - may not even need this state once this is implemented
            self.move_pub.publish(Twist())
            return

        if self.running:
            new_image = np.array(cv_image)
            self.image_difference = np.mean((new_image - self.last_image) ** 2 )
            # print("Image difference:", self.image_difference)
            if self.image_difference < RESPAWN_THRESHOLD:
                self.spawn_position(HOME)
                self.state.current_state = self.state.DRIVING
                self.state.set_location(State.Location.ROAD)
                print("Respawned")
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

        # elif action == State.Action.CRY:
        #     # TODO: implement pedestrian state
        #     self.driving(cv_image)
        # elif action == State.Action.RESPAWN:
        #     # TODO: implement desert state
        #     self.driving(cv_image)

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

        # If the clue has not improved in 3 seconds, find the letters, and reset the max area
        if time.time() - self.state.clue_improved_time > 2:
            print("CLUE IS GOOD")
            self.submit_clue()
            self.state.clue_improved_time = time.time()+300 # Set the time after comp, so it doesn't submit again
            self.state.max_area = 0
            

        # Generating masks to check for certain colurs
        # Pink
        pink_mask = cv2.inRange(cv_image, LOWER_PINK, UPPER_PINK)
        pink_pixel_count = cv2.countNonZero(pink_mask)
        # print("Pink pixel count:", pink_pixel_count)
        # Red
        red_mask = cv2.inRange(cv_image, LOWER_RED, UPPER_RED)
        red_pixel_count = cv2.countNonZero(red_mask)
        # print("Red pixel count:", red_pixel_count)

        # Create a local state that will be added by bitwise OR
        state = 0b00000000
        
        if red_pixel_count > RED_THRESHOLD:
            state |= self.state.RED

        if pink_pixel_count > PINK_THRESHOLD:
            state |= self.state.PINK
            if (self.state.last_state & self.state.PINK) == 0:
                state |= self.state.PINK_ON

        state |= self.state.DRIVING

        self.state.last_state = self.state.current_state
        self.state.current_state = state

        # print("state:", self.state.current_state)
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

        cv2.imshow("Driving Image", cv_image)
        cv2.waitKey(3)

        # PID controller
        move = Twist()
        
        # print("Road error", error)
        derivative = error - self.previous_error
        self.previous_error = error

        move.angular.z = -(KP * error + KD * derivative)
        # print("Road angular speed:", move.angular.z)

        # decrease linear speed as angular speed increases from a max of 3 down to 1.xx? if abs(error) > 400
        move.linear.x = max(0, MAX_SPEED - SPEED_DROP * abs(error))
        # print("Road linear speed:", move.linear.x)

        self.move_pub.publish(move)

    def offroad_driving(self, cv_image):
        lost_left = False
        lost_right = False
        contour_colour = (0, 0, 255)

        top_2_contours = self.process_image(cv_image)
        cv2.drawContours(cv_image, top_2_contours, -1, contour_colour, 2)

        centroids = []
        if len(top_2_contours) != 0:
        # get the centroids of the two largest contours
            for i in range(len(top_2_contours)):
                M = cv2.moments(top_2_contours[i])
                if M["m00"] == 0:
                    return
                centroids.append(int(M["m10"] / M["m00"]))

            if centroids[0] < IMAGE_WIDTH // 2 and centroids[1] < IMAGE_WIDTH // 2:
                lost_right = True
            elif centroids[0] > IMAGE_WIDTH // 2 and centroids[1] > IMAGE_WIDTH // 2:
                lost_left = True

            # positive means the car should turn left
            error = IMAGE_WIDTH / 2 - np.mean(centroids)
        else:
            error = -IMAGE_WIDTH / 2 if self.previous_error < 0 else IMAGE_WIDTH / 2

        # TODO: maybe smooth this out? might not need it idk
        if lost_left:  # This will bias the car to the left if the left side is lost
            error = LOSS_FACTOR
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
        if lost_right:
            error = -LOSS_FACTOR
            # write onto image
            cv2.putText(
                cv_image,
                "Lost RIGHT",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # red circles showing the centroids
        cv2.circle(
            cv_image, (centroids[0], IMAGE_HEIGHT - CROP_AMOUNT), 5, (0, 0, 255), -1
        )
        cv2.circle(
            cv_image, (centroids[1], IMAGE_HEIGHT - CROP_AMOUNT), 5, (0, 0, 255), -1
        )
        # green circle showing the middle of the image
        cv2.circle(
            cv_image, (IMAGE_WIDTH // 2, IMAGE_HEIGHT - CROP_AMOUNT), 5, (0, 255, 0), -1
        )
        # blue circle showing the mean of the centroids
        cv2.circle(
            cv_image,
            (int(np.mean(centroids)), IMAGE_HEIGHT - CROP_AMOUNT),
            5,
            (255, 0, 0),
            -1,
        )

        # cv2.imshow("Desert Mask", mask)
        cv2.imshow("Desert Image", cv_image)
        cv2.waitKey(3)

        # PID controller
        move = Twist()

        # print("Desert error", error)
        derivative = error - self.previous_error
        self.previous_error = error

        move.angular.z = DKP * KP * error + KD * derivative
        # print("Desert angular speed:", move.angular.z)

        move.linear.x = max(0, MAX_SPEED - SPEED_DROP * abs(error))
        # print("Desert linear speed:", move.linear.x)

        self.move_pub.publish(move)

    # TODO: make this more efficient/ more reliable
    def process_image(self, image):
        # mask the image
        mask = cv2.inRange(image, LOWER_DIRT, UPPER_DIRT)

        # Define a kernel (structuring element)
        kernel = np.ones((3, 3), np.uint8)

        # Perform 1st dilation
        dilated_image = cv2.dilate(mask, kernel, iterations=1)

        # Perform erosion
        eroded_image = cv2.erode(dilated_image, kernel, iterations=2)

        # Perform 2nd dilation
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=8)

        dilate_contours, _ = cv2.findContours(
            dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        dilate_contours = sorted(dilate_contours, key=cv2.contourArea, reverse=True)

        top_two_contours = dilate_contours[:2]

        return top_two_contours

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
        lower_hsv = np.array([lh,ls,lv])
        upper_hsv = np.array([uh,us,uv])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # find contours in the mask
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)

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

            # THE FOLLOWING HAPPENS IF A CLUE IS FOUND
            # If the area is bigger than the previous biggest area, update the best clue
            # and reset the time since the clue improved
            if clue_size > self.state.max_area:

                self.state.max_area = clue_size
                self.state.clue_improved_time = time.time()
            
                # Template that the clue will be warped onto
                clue = np.zeros((400, 600, 3), np.uint8)

                top_right, bottom_right, bottom_left, top_left = self.find_clue_corners(clue_border)

                input_pts = np.float32([top_right, bottom_right, bottom_left, top_left])
                output_pts = np.float32([[clue.shape[1],0], [clue.shape[1],clue.shape[0]], [0,clue.shape[0]], [0,0]])

                matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
                self.state.best_clue = cv2.warpPerspective(cv_image,matrix,(clue.shape[1], clue.shape[0]),flags=cv2.INTER_LINEAR)
        
        return

    def submit_clue(self):
        # Get the words from the clue 
        type, clue = self.find_words()
        print("TYPE:", type)
        print("CLUE:", clue)

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

    def find_words(self):
        print("\n\n\n\n\n\n\n\n\n\n\nFINDING WORDS")
        clue = self.state.best_clue

        cv2.imshow("Clue", clue)
        cv2.waitKey(3)

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

        lower_hsv = np.array([lh,ls,lv])
        upper_hsv = np.array([uh,us,uv])

        type_mask = cv2.inRange(type_hsv, lower_hsv, upper_hsv)
        clue_mask = cv2.inRange(clue_hsv, lower_hsv, upper_hsv)

        # Dilate then erode to fill them in (?)
        kernel = np.ones((3,3),np.uint8)
        for i in range(10):
            type_mask = cv2.dilate(type_mask, kernel, iterations=1)
            clue_mask = cv2.dilate(clue_mask, kernel, iterations=1)

            type_mask = cv2.erode(type_mask, kernel, iterations=1)
            clue_mask = cv2.erode(clue_mask, kernel, iterations=1)


        # Find contours (letters)

        type_cnts, _ = cv2.findContours(type_mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

        clue_cnts, _ = cv2.findContours(clue_mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

        # Remove contours that are too small

        type_cnts = [c for c in type_cnts if cv2.contourArea(c) > 100]
        clue_cnts = [c for c in clue_cnts if cv2.contourArea(c) > 100]

        # The bounding boxes of the contours (a box with a letter in it)
        clueBoundingBoxes = [cv2.boundingRect(c) for c in clue_cnts]
        typeBoundingBoxes = [cv2.boundingRect(c) for c in type_cnts]

        # sort the contours from left-to-right
        typeBoundingBoxes = sorted((typeBoundingBoxes), key=lambda b:b[0], reverse=False)
        clueBoundingBoxes = sorted((clueBoundingBoxes), key=lambda b:b[0], reverse=False)
        
        # If any bounding boxes are too big, split it (in case 2 letters' contours are connected)
        for i in range(len(clueBoundingBoxes)):
            if clueBoundingBoxes[i][2] > 60:
                box1 = (clueBoundingBoxes[i][0], clueBoundingBoxes[i][1], clueBoundingBoxes[i][2]//2, clueBoundingBoxes[i][3])
                box2 = (clueBoundingBoxes[i][0] + clueBoundingBoxes[i][2]//2, clueBoundingBoxes[i][1], clueBoundingBoxes[i][2]//2, clueBoundingBoxes[i][3])
                clueBoundingBoxes.remove(clueBoundingBoxes[i])
                clueBoundingBoxes.insert(i, box1)
                clueBoundingBoxes.insert(i+1, box2)
                print("SPLIT LETTERS!")

        for i in range(len(typeBoundingBoxes)):
            if typeBoundingBoxes[i][2] > 60:
                box1 = (typeBoundingBoxes[i][0], typeBoundingBoxes[i][1], typeBoundingBoxes[i][2]//2, typeBoundingBoxes[i][3])
                box2 = (typeBoundingBoxes[i][0] + typeBoundingBoxes[i][2]//2, typeBoundingBoxes[i][1], typeBoundingBoxes[i][2]//2, typeBoundingBoxes[i][3])
                typeBoundingBoxes.remove(typeBoundingBoxes[i])
                typeBoundingBoxes.insert(i, box1)
                typeBoundingBoxes.insert(i+1, box2)
                print("SPLIT LETTERS!")

        # Letter Template
        letter = np.zeros((130, 80, 3), np.uint8)

        letter_imgs = []

        # For each rectangle (which is a letter), warp perspective to get the letter into the template
        for i in range(len(typeBoundingBoxes)):
            (x, y, w, h) = typeBoundingBoxes[i]
            input_pts = np.float32([[x+w,y], [x+w,y+h], [x,y+h], [x,y]])
            output_pts = np.float32([[letter.shape[1],0], [letter.shape[1],letter.shape[0]], [0,letter.shape[0]], [0,0]])
            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
            letter = cv2.warpPerspective(type_plate,matrix,(letter.shape[1], letter.shape[0]),flags=cv2.INTER_LINEAR)
            letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
            letter_imgs.append(np.array(letter))

        for i in range(len(clueBoundingBoxes)):
            (x, y, w, h) = clueBoundingBoxes[i]
            input_pts = np.float32([[x+w,y], [x+w,y+h], [x,y+h], [x,y]])
            output_pts = np.float32([[letter.shape[1],0], [letter.shape[1],letter.shape[0]], [0,letter.shape[0]], [0,0]])
            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
            letter = cv2.warpPerspective(clue_plate,matrix,(letter.shape[1], letter.shape[0]),flags=cv2.INTER_LINEAR)
            letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
            letter_imgs.append(np.array(letter))

        letter_imgs = np.expand_dims(np.array(letter_imgs), axis=-1)

        # Now that there are images, input them into the neural network to get the string
        print("Found Letters, beginning prediction")
        start_time = time.time()

        prediction_arrays = self.letter_model.predict(letter_imgs)

        predicted_values = np.argmax(prediction_arrays, axis=1)
        # turn the predicted values into the characters/numbers
        predicted_values = [chr(x+65) if x < 26 else str(x-26) for x in predicted_values]
        predicted_values = "".join(predicted_values)

        type = predicted_values[:len(typeBoundingBoxes)]
        clue = predicted_values[len(typeBoundingBoxes):]

        print("Completed Prediction. Took " + str(time.time()-start_time) + " seconds")

        return type, clue


    def find_clue_corners(self, clue_border):
        # Simplify the contour to 4 points
        epsilon = 0.1*cv2.arcLength(clue_border,True)
        approx = cv2.approxPolyDP(clue_border,epsilon,True)

        while len(approx) != 4:
            if len(approx) > 4:
                epsilon = epsilon * 1.05
                approx = cv2.approxPolyDP(clue_border,epsilon,True)
            elif len(approx) < 4:
                epsilon = epsilon * 0.96
                approx = cv2.approxPolyDP(clue_border,epsilon,True)

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



    # TODO: Implement this when we figure out what to do/ if we figure it out...
    def pedestrian(self, cv_image):
        """Function for the PEDESTRIAN state"""
        start_time = time.time()
        move = Twist()
        move.angular.z = 0.0
        move.linear.x = 0.1
        # while time.time() - start_time < 3:
            # self.move_pub.publish(move)


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
