# !/usr/bin/env python

import cv2

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image,CompressedImage
import numpy as np
import message_filters
import rospy
data_complete=np.array(0)
first=True
def callback(image , motor_data):
    #print("im here")
    global data_complete
    global first
    brige = CvBridge()

    try:
        # brige.
        # frame=brige.imgmsg_to_cv2(image, "bgr8")
        frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
        # frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)

    # (rows, cols, channels) = cv_image.shape
    # if cols > 60 and rows > 60:
    #     cv2.circle(cv_image, (50, 50), 10, 255)
    if first:
        data_complete = get_array_image_steering(frame,motor_data.point.z)
        first=False
    else:
        data = get_array_image_steering(frame,motor_data.point.z)
        data_complete = np.vstack([data_complete, data])



def get_array_image_steering(image,angle):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(gray_image).flatten()
    cv2.imshow("Image Window", gray_image)
    cv2.waitKey(1)
    return np.append(angle, image)






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
    global data_complete
    print(data_complete,first)
    np.savetxt("grey_image_data.csv", data_complete, delimiter=",", fmt='%.6f')


if __name__ == '__main__':
    listener()
