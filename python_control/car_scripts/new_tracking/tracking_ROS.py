# !/usr/bin/env python
from tracking_performance_class import tracking_red_dots
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CompressedImage  # Image
import cv2
import rospy
import numpy as np

rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=0)
rate = rospy.Rate(25)  # Frequenz der Anwendung
x_0 = 326
y_0 = 396
x_1 = 400
y_1 = 398


def callback(image, tracker):
    # print(image.encoding)
    brige = CvBridge()
    try:
        # frame = brige.imgmsg_to_cv2(image, "passthrough")
        frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    global x_0, y_0, x_1, y_1
    x_0, y_0, x_1, y_1, can_see = tracker.get_red_pos(frame, x_0, y_0, x_1, y_1)

    #cv2.imshow("Image Window", frame)
    #cv2.waitKey(1)
    circle = cv2.circle(frame, (x_1, y_1), 5, 120, -1)
    circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
    cv2.imshow("Image Window", circle)
    cv2.waitKey(1)

    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = 0  # aktuell in tick rate(+- 3900)
    message.point.y = 2  # not used
    message.point.z = 0  # in grad(max +-20)
    rospy.loginfo(message)
    pub.publish(message)
    rate.sleep()
    # cv2.waitKey()


def listener():
    # tracker = tracking_red_dots(308,410)
    # tracker = tracking_red_dots(960, 1280,350,900,400,960)
    tracker = tracking_red_dots(576, 768)

    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback, tracker)
    rospy.spin()
    # video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
