#!/usr/bin/env python
# license removed for brevity
import time
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from pyquaternion import Quaternion
from nav_msgs.msg import Odometry

import time

def talker(odometry_data):
    global size,k,array_to_save
    x_position_car = odometry_data.pose.pose.position.x
    y_position_car = odometry_data.pose.pose.position.y
    array_to_save[k, 0] = x_position_car
    array_to_save[k, 1] = y_position_car
    print(k)
    if k==size-1:
        print(k)
        array_tmp=array_to_save
        array_to_save = np.zeros((size*2,2))
        array_to_save[0:k+1,:]=array_tmp
        size=size*2
    k = k + 1

def dxl_control():
    rospy.Subscriber('odometry_car', Odometry, talker)

    #rate.sleep()
    # pos_drive_to()
    # rospy.spin()
    rospy.spin()




if __name__ == '__main__':
    try:
        dxl_control()
    except rospy.ROSInterruptException:
        pass



