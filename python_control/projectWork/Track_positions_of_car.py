#!/usr/bin/env python
# license removed for brevity
import time
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from pyquaternion import Quaternion
from nav_msgs.msg import Odometry

import time


x_position = 0
y_position = 0

rospy.init_node('subscriber', anonymous=True)
Kv = 0.5
Kh = 1
Kacell = 10
saved_steering = 0
# tetta_car_ofset = 40
transform_angle_front=0.8156*180/3.14159
tetta_car_ofset = transform_angle_front+6.268
scaling_transform_axes=2.764

def rotation_2d(x, y, angle):
    x2 = x * np.cos(angle) - np.sin(angle) * y
    y2 = x * np.sin(angle) + np.cos(angle) * y
    return x2, y2

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def radToDegree(tmp):
    return tmp / np.pi * 180


def degreeToRad(tmp):
    return tmp * np.pi / 180


def maxValue(regulate, max_value):
    if regulate > max_value:
        regulate = max_value
    if regulate < -max_value:
        regulate = -max_value
    return regulate


def angularDiff(a, b):
    diff = a - b
    if (diff < -np.pi):
        diff = diff + np.pi * 2
    if (diff > np.pi):
        diff = diff - np.pi * 2
    return diff

size=4
array_to_save=np.zeros((4,3))
k=0



def talker(odometry_data):
    global size,k,array_to_save
    my_quaternion = Quaternion(np.asarray(
        [odometry_data.pose.pose.orientation.x, odometry_data.pose.pose.orientation.y,
         odometry_data.pose.pose.orientation.z,
         odometry_data.pose.pose.orientation.w]))
    theta_car = rotationMatrixToEulerAngles(my_quaternion.rotation_matrix)[0] + degreeToRad(tetta_car_ofset)
    distance = odometry_data.pose.pose.position.z
    x_offset_car, y_offset_car = rotation_2d(1, 0, theta_car)
    x_offset_car = x_offset_car * distance * scaling_transform_axes
    y_offset_car = y_offset_car * distance * scaling_transform_axes
    x_position_car = odometry_data.pose.pose.position.x + x_offset_car
    y_position_car = odometry_data.pose.pose.position.y + y_offset_car
    array_to_save[k, 0] = x_position_car
    array_to_save[k, 1] = y_position_car
    array_to_save[k, 2] = theta_car
    print(k)
    if k==size-1:
        print(k)
        array_tmp=array_to_save
        array_to_save = np.zeros((size*2,3))
        array_to_save[0:k+1,:]=array_tmp
        size=size*2
    k = k + 1

def dxl_control():
    rospy.Subscriber('odometry_car', Odometry, talker)
    while not rospy.is_shutdown():
        np.save("file.npy", array_to_save)

    #rate.sleep()
    # pos_drive_to()
    # rospy.spin()
    rospy.spin()




if __name__ == '__main__':
    try:
        dxl_control()
    except rospy.ROSInterruptException:
        pass

