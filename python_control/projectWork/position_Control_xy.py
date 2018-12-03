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


def setPos(odometry_data, x, y):  # has to be queed every step

    my_quaternion = Quaternion(np.asarray(
        [odometry_data.pose.pose.orientation.x, odometry_data.pose.pose.orientation.y,
         odometry_data.pose.pose.orientation.z,
         odometry_data.pose.pose.orientation.w]))
    theta_car = rotationMatrixToEulerAngles(my_quaternion.rotation_matrix)[0]+degreeToRad(tetta_car_ofset)

    distance = odometry_data.pose.pose.position.z
    x_offset_car, y_offset_car = rotation_2d(1, 0, theta_car) * distance * 38.0/12.9
    x_position_car = odometry_data.pose.pose.position.x + x_offset_car
    y_position_car = odometry_data.pose.pose.position.y + y_offset_car

    theta_wanted = np.arctan2((y - y_position_car), (x - x_position_car))
    print(theta_wanted)
    print(theta_car)
    gamma = Kh * (angularDiff(theta_wanted, theta_car))
    gamma = maxValue(gamma * 180 / 3.14159, 29)
    print(gamma)
    steering = gamma

    # saved_steering = gamma
    v_wanted = Kv * np.sqrt(
        (y - y_position_car) * (y - y_position_car) + (
                x - x_position_car) * (x - x_position_car))
    # v_wanted = maxValue(v_wanted, 4095)
    # accell_in = Kacell * (v_wanted - speed_car)
    accell_in = maxValue(Kacell * v_wanted, 4000)

    return (steering, accell_in)


x_position = 0
y_position = 0

rospy.init_node('subscriber', anonymous=True)
pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=0)
rate = rospy.Rate(150)  # Frequenz der Anwendung

Kv = 0.5
Kh = 1
Kacell = 10
saved_steering = 0
# tetta_car_ofset = 40
tetta_car_ofset = -40


def talker(odometry_data):
    start = time.time()
    global x_position, y_position
    # odometry= Odometry()
    # odometry.pose.pose.orientation.x
    steering, accell_in = setPos(odometry_data, 0, 0)

    # accell_in=0
    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = accell_in  # aktuell in tick rate(+- 3900)
    message.point.y = 2  # not used
    message.point.z = steering  # in grad(max +-20)
    rospy.loginfo(message)
    pub.publish(message)
    diff = time.time() - start
    print(diff)
    rate.sleep()


def pos_drive_to():
    # global x_position,y_position
    while 1:
        x_pos = raw_input('X Position:')
        try:
            x_position = float(x_pos)
            break
        except ValueError:
            print('should have used a number')
            continue
    while 1:
        y_pos = raw_input('Y Position:')
        try:
            y_position = float(y_pos)
            break
        except ValueError:
            print('should have used a number')
            continue


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
