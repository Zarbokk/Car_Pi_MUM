# !/usr/bin/env python

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CompressedImage
import message_filters
import rospy



rospy.init_node('listener', anonymous=True)
rate = rospy.Rate(20)  # Frequenz der Anwendung


def callback(image):
    brige = CvBridge()
    try:
        frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)

    cv2.imshow("Image Window", frame)
    cv2.waitKey(1)
    rate.sleep()

def listener():

    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback)
    rospy.spin()
    #video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    listener()
