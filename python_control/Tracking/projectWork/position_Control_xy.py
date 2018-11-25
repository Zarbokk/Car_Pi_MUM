#!/usr/bin/env python
# license removed for brevity
import time
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from pyquaternion import Quaternion
from navigation.msg import Odometry

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
    return tmp/np.pi*180
def degreeToRad(tmp):
    return tmp*np.pi/180
def maxValue(regulate,max_value):
    if regulate>max_value:
        regulate=max_value
    if regulate<-max_value:
        regulate=-max_value
    return regulate
def angularDiff(a,b):
    diff = a - b
    if (diff < -np.pi):
        diff = diff + np.pi * 2
    if (diff > np.pi):
        diff = diff - np.pi * 2
    return diff
def setPos(odometry_data,x,y):  # has to be queed every step
    my_quaternion = Quaternion(np.asarray(
        [odometry_data.orientation.x_val, odometry_data.orientation.y_val, odometry_data.orientation.z_val,
         odometry_data.orientation.w_val]))
    tetta_car = rotationMatrixToEulerAngles(my_quaternion.rotation_matrix)[0]
    # print(rotationMatrixToEulerAngles(my_quaternion.rotation_matrix))

    # tetta_wanted = math.atan((wanted_pos[1]-car_pos.position.y_val)/(wanted_pos[0]-car_pos.position.x_val))
    tetta_wanted = np.arctan2((y - odometry_data.y_val), (x - odometry_data.x_val))
    # print(tetta_wanted)
    # print(tetta_car)
    gamma = Kh * (angularDiff(tetta_wanted, tetta_car+tetta_car_ofset))
    # print(gamma)
    gamma = maxValue(gamma, 30)

    steering = gamma

    #saved_steering = gamma
    v_wanted = Kv * np.sqrt(
        (y - odometry_data.orientation.y_val) * (y - odometry_data.orientation.y_val) + (
                x - odometry_data.orientation.x_val) * (x - odometry_data.orientation.x_val))
    #v_wanted = maxValue(v_wanted, 4095)
    #accell_in = Kacell * (v_wanted - speed_car)
    accell_in = maxValue(Kacell*v_wanted, 4095)
    return(steering,accell_in)




rospy.init_node('talker', anonymous=True)
pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=10)
rate = rospy.Rate(30)  # Frequenz der Anwendung

Kv = 0.5
Kh = 4.
Kacell = 10
saved_steering = 0
tetta_car_ofset=10

def talker(odometry_data):


    input_motor_speed=0

    steering, accell_in =setPos(odometry_data,1000,500)



    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x=accell_in#aktuell in tick rate(+- 3900)
    message.point.y=2#not used
    message.point.z=steering#in grad(max +-20)
    rospy.loginfo(message)
    pub.publish(message)
    rate.sleep()


def dxl_control():
    rospy.init_node('subscriber', anonymous=True)
    rospy.Subscriber('car_position', Odometry, talker)
    rospy.spin()

if __name__ == '__main__':
    try:
        dxl_control()
    except rospy.ROSInterruptException:
        pass
