#! /usr/bin/env python3

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

CAPTURE_PATH = "/home/fizzer/ros_ws/src/enph353_competition_controller/image_collection/collected_images/"


class topic_publisher:
    def __init__(self):
        self.image_sub = rospy.Subscriber(
            "R1/pi_camera/image_raw", Image, self.callback
        )
        self.bridge = CvBridge()
        self.img_count = 0
        print("Initialized")

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imshow("Image", image)
        cv2.waitKey(3)
        capture_rate = 30
        capture_start = 150
        if self.img_count % capture_rate == 0 and self.img_count > capture_start:
            save_name = self.img_count // capture_rate - capture_start // capture_rate
            cv2.imwrite(CAPTURE_PATH + "mountain_img%d.png" % save_name, image)
            print("Saved image %d" % save_name)
        self.img_count += 1
        print(self.img_count)


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
