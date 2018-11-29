from red_dots_tracking_class import tracking_red_dots
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import cv2
import rospy
import numpy as np
#video = cv2.VideoCapture("/home/tim/Dokumente/1280_32.avi")
#ok, frame = video.read()
#tracker = tracking_red_dots(np.size(frame, 0),np.size(frame, 1))

#while ok:
#

#    x_0, y_0, x_1, y_1 = tracker.get_red_pos(frame)#

#    circle= cv2.circle(frame, (x_1, y_1), 5, 120, -1)
#    circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
#    cv2.imshow('largest contour',circle)
#    cv2.waitKey()
#    ok, frame = video.read()








# !/usr/bin/env python



def callback(image,tracker):
    brige = CvBridge()
    try:
        frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    #cv2.imshow("Image Window", frame)
    #cv2.waitKey(1)

    #cv2.waitKey()


    #video.write(cv_image)
    #print("huhu234")
    x_0, y_0, x_1, y_1 = tracker.get_red_pos(frame)
    circle= cv2.circle(frame, (x_1, y_1), 5, 120, -1)
    circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
    #cv2.destroyAllWindows()
    #video.release()
    # (rows, cols, channels) = cv_image.shape
    # if cols > 60 and rows > 60:
    #     cv2.circle(cv_image, (50, 50), 10, 255)
    #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image Window", circle)
    cv2.waitKey(1)
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
    #tracker = tracking_red_dots(308,410)
    tracker = tracking_red_dots(960, 1280)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video = cv2.VideoWriter('/home/tim/Dokumente/Video_car_find.avi', fourcc, 20.0, (1280, 960))

    #video=3
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback, tracker)
    rospy.spin()
    #video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
