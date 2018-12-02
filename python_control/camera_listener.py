
# !/usr/bin/env python

import cv2

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CompressedImage

import message_filters
import rospy

def callback(image , motor_data):
    brige = CvBridge()
    try:
        cv_image = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)

    # (rows, cols, channels) = cv_image.shape
    # if cols > 60 and rows > 60:
    #     cv2.circle(cv_image, (50, 50), 10, 255)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image Window", gray_image)
    cv2.waitKey(3)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    image_sub = message_filters.Subscriber('/raspicam_node/image/compressed', CompressedImage)
    car_sub = message_filters.Subscriber('/car_motor_input', PointStamped)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, car_sub], 10,1)
    ts.registerCallback(callback)
    rospy.init_node('listener', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    listener()
