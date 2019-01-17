# !/usr/bin/env python

import cv2

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
import numpy as np
import message_filters
import rospy
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from tracking_performance_class import tracking_red_dots

rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('car_input_03', PointStamped, queue_size=1)
rate = rospy.Rate(20)  # Frequenz der Anwendung
ueberholen = False
linie_folgen = True
fahrzeug_folgen = False
x_0 = 326
y_0 = 396
x_1 = 400
y_1 = 398
alpha_car = 0
x_car = 0
y_car = 0
tracker = tracking_red_dots(576, 768)

time_following = 0


def transform_to_opencv(image):
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    return image


def tiefpass(x, x_old, rate=0.5):
    return x * (1 - rate) + x_old * rate


def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def maxValue(regulate, max_value):
    if regulate > max_value:
        regulate = max_value
    if regulate < -max_value:
        regulate = -max_value
    return regulate


def controller_verfolgen(x, y, alpha):
    # print("distance x:", x, "distance y:", y, alpha * 180 / np.pi, distance)
    x = x - 350

    a = -y / x ** 3 + np.tan(alpha) / x ** 2
    b = y / x ** 2 - a * x
    y_pos = cubic_function(x / 2, a, b, 0, 0)

    theta_wanted = np.arctan2(y_pos, x / 2)
    # print(x / 2, y_pos, theta_wanted * 180 / np.pi)
    # print(theta_wanted)
    # print(theta_car)
    gamma = 1 * (theta_wanted)
    gamma = maxValue(gamma * 180 / 3.14159, 29)
    # print(gamma)
    steering = gamma

    # saved_steering = gamma
    v_wanted = 1 * (x / 2 - 50)
    # v_wanted = maxValue(v_wanted, 4095)
    # accell_in = Kacell * (v_wanted - speed_car)
    accell_in = maxValue(9 * v_wanted, 4000)
    return accell_in, steering


def overtake(imu_03_sub, car_f_sub, imu_10_sub):
    return 0, 0


def follow_line(frame, ultraschall_sub, vel=0.6):
    if ultraschall_sub.range < 15:
        global linie_folgen, fahrzeug_folgen, x_0, x_1, y_0, y_1, alpha_car, time_following
        linie_folgen = False
        fahrzeug_folgen = True
        x_0 = 326
        y_0 = 396
        x_1 = 400
        y_1 = 398
        alpha_car = 0
        time_following = 0
    return vel * 4000 / 2.2, 0


def follow_car(frame, tracker):
    global x_0, y_0, x_1, y_1, alpha_car, x_car, y_car, time_following, ueberholen, fahrzeug_folgen
    x_0_old = x_0
    y_0_old = y_0
    x_1_old = x_1
    y_1_old = y_1
    alpha_old = alpha_car
    x_0, y_0, x_1, y_1, can_see = tracker.get_red_pos(frame, x_0, y_0, x_1, y_1)
    x_0 = tiefpass(x_0, x_0_old)
    y_0 = tiefpass(y_0, y_0_old)
    x_1 = tiefpass(x_1, x_1_old)
    y_1 = tiefpass(y_1, y_1_old)

    drehung = -np.arctan2(y_1 - y_0, x_1 - x_0) * 2.9
    drehung = drehung * 180 / np.pi
    # print("drehung", float(drehung))
    fov = 62.2
    f = 592.61
    hoehe_cam = 220
    de = 73
    dp = x_1 - x_0
    alpha_car = -((x_1 + x_0) / 2 - 768 / 2) / (768 / 2) * fov / 2 / 180 * np.pi * 1.08
    alpha_car = tiefpass(alpha_car, alpha_old, 0.8)
    distance = de * f / dp * (1 + abs(alpha_car) / 4 * 1.1) * (1 - abs(drehung / 100) / 2)
    distance = np.sqrt(distance ** 2 - hoehe_cam ** 2)
    x_car = np.cos(alpha_car) * distance
    y_car = np.sin(alpha_car) * distance
    accell_in, steering = controller_verfolgen(x_car, y_car, alpha_car)

    if time_following > 2:
        ueberholen = True
        fahrzeug_folgen = False

    return accell_in, steering


def callback(image_sub=Image, imu_03_sub=Imu, ultraschall_sub=Range, car_f_sub=PointStamped, imu_10_sub=Imu):
    frame = transform_to_opencv(image_sub)
    global ueberholen, linie_folgen, fahrzeug_folgen, time_following

    if ueberholen:
        accell_in, steering = overtake(imu_03_sub, car_f_sub, imu_10_sub)
    elif linie_folgen:
        accell_in, steering = follow_line(frame, ultraschall_sub)
    elif fahrzeug_folgen:
        accell_in, steering = follow_car(frame)
    time_following = time_following + 1 / 20

    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = accell_in  # aktuell in tick rate(+- 3900)
    message.point.y = 2  # not used
    message.point.z = steering  # in grad(max +-20)
    # rospy.loginfo(message)
    pub.publish(message)
    rate.sleep()


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    image_sub = message_filters.Subscriber('/raspicam_node/image', Image)
    ultraschall_sub = message_filters.Subscriber('/distance_sensor_03', Range)
    imu_03_sub = message_filters.Subscriber('/IMU_03', Imu)
    imu_10_sub = message_filters.Subscriber('/IMU_10', Imu)
    car_f_sub = message_filters.Subscriber('/car_input_10', PointStamped)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, imu_03_sub, ultraschall_sub, car_f_sub, imu_10_sub],
                                                     10, 2)
    ts.registerCallback(callback)
    # rospy.init_node('listener', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    listener()
