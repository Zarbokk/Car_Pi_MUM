
# !/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CompressedImage
import cv2
import message_filters
import rospy

def callback(image,video):
    brige = CvBridge()
    try:
        cv_image = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    video.write(cv_image)
    #cv2.destroyAllWindows()
    #video.release()
    # (rows, cols, channels) = cv_image.shape
    # if cols > 60 and rows > 60:
    #     cv2.circle(cv_image, (50, 50), 10, 255)
    #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Image Window", cv_image)

    #cv2.waitKey()


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # 1280
    # x960
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('/home/tim/Documents/Car_Pi_MUM/python_control/ParameterIdent/calibration_data.avi', fourcc, 20.0, (768, 576))
    #video=3
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback, video)
    rospy.spin()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
