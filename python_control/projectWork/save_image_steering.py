# !/usr/bin/env python

import cv2

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image,CompressedImage
import numpy as np
import message_filters
import rospy


n_rows=10
k=0
i=0
data_complete=np.ones((n_rows,410*308+1))
def callback(image , motor_data):
    #print("im here")
    global data_complete
    global k,i,n_rows
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


    data_complete[k,:] = get_array_image_steering(frame,motor_data.point.z)
    k=k+1
    if k==n_rows:
        k=0
        np.savetxt("training_NN/train_data/grey_image_data_backward{0}.csv".format(i), data_complete, delimiter=",", fmt='%.6f')
        i=i+1



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
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, car_sub], 10,2)
    ts.registerCallback(callback)
    rospy.init_node('listener', anonymous=True)
    rospy.spin()
    global data_complete



if __name__ == '__main__':
    listener()
