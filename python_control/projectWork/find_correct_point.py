#!/usr/bin/env python
# license removed for brevity
import time
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from pyquaternion import Quaternion
from nav_msgs.msg import Odometry

import time


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



x_position = 0
y_position = 0

rospy.init_node('subscriber', anonymous=True)
tetta_car_ofset = 6.268
transform_angle_front=0.8156*180/3.14159

def talker(odometry_data):
    my_quaternion = Quaternion(np.asarray(
        [odometry_data.pose.pose.orientation.x, odometry_data.pose.pose.orientation.y,
         odometry_data.pose.pose.orientation.z,
         odometry_data.pose.pose.orientation.w]))
    theta_car = rotationMatrixToEulerAngles(my_quaternion.rotation_matrix)[0] + degreeToRad(tetta_car_ofset)
    #0.925403243329828 angle streat
    distance = odometry_data.pose.pose.position.z
    x_offset_car, y_offset_car = rotation_2d(1, 0, theta_car + degreeToRad(0.8156722069084836))
    x_offset_car = x_offset_car * distance * 2.764
    y_offset_car = y_offset_car * distance * 2.764
    x_position_car = odometry_data.pose.pose.position.x  + x_offset_car
    y_position_car = odometry_data.pose.pose.position.y  + y_offset_car


    print(distance)
    print(rotationMatrixToEulerAngles(my_quaternion.rotation_matrix)[0])
    print("position odometry output",odometry_data.pose.pose.position.x,odometry_data.pose.pose.position.y)
    print("position of front Point:",x_position_car,y_position_car)



def dxl_control():
    rospy.Subscriber('odometry_car', Odometry, talker)
    # while not rospy.is_shutdown():
    # rate.sleep()
    # pos_drive_to()
    # rospy.spin()
    rospy.spin()


if __name__ == '__main__':
    try:
        dxl_control()
    except rospy.ROSInterruptException:
        pass
