# !/usr/bin/env python
from tracking_performance_class import tracking_red_dots
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image  # CompressedImage  # Image
import cv2
import rospy
import numpy as np

rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('car_input_03', PointStamped, queue_size=1)
rate = rospy.Rate(25)  # Frequenz der Anwendung
x_0 = 326
y_0 = 396
x_1 = 400
y_1 = 398
alpha = 0


def maxValue(regulate, max_value):
    if regulate > max_value:
        regulate = max_value
    if regulate < -max_value:
        regulate = -max_value
    return regulate


def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def tiefpass(x, x_old, rate=0.5):
    return x * (1 - rate) + x_old * rate

def controller_verfolgen(x,y,alpha):

    #print("distance x:", x, "distance y:", y, alpha * 180 / np.pi, distance)
    x = x - 350

    a = -y / x ** 3 + np.tan(alpha) / x ** 2
    b = y / x ** 2 - a * x
    y_pos = cubic_function(x / 2, a, b, 0, 0)

    theta_wanted = np.arctan2(y_pos, x / 2)
    #print(x / 2, y_pos, theta_wanted * 180 / np.pi)
    # print(theta_wanted)
    # print(theta_car)
    gamma = 1 * (theta_wanted)
    gamma = maxValue(gamma * 180 / 3.14159, 29)
    # print(gamma)
    steering = gamma

    # saved_steering = gamma
    v_wanted = 1 * (x/2-50)
    # v_wanted = maxValue(v_wanted, 4095)
    # accell_in = Kacell * (v_wanted - speed_car)
    accell_in = maxValue(9 * v_wanted, 4000)
    return accell_in, steering

def callback(image, tracker):
    # print(image.encoding)
    brige = CvBridge()
    try:
        frame = brige.imgmsg_to_cv2(image, "passthrough")
        # frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    global x_0, y_0, x_1, y_1, alpha
    x_0_old = x_0
    y_0_old = y_0
    x_1_old = x_1
    y_1_old = y_1
    x_0, y_0, x_1, y_1, can_see = tracker.get_red_pos(frame, x_0, y_0, x_1, y_1)
    x_0 = tiefpass(x_0, x_0_old)
    y_0 = tiefpass(y_0, y_0_old)
    x_1 = tiefpass(x_1, x_1_old)
    y_1 = tiefpass(y_1, y_1_old)
    alpha_old = alpha

    drehung = -np.arctan2(y_1 - y_0, x_1 - x_0) * 2.9
    drehung = drehung * 180 / np.pi
    #print("drehung", float(drehung))
    fov = 62.2
    f = 592.61
    hoehe_cam = 220
    de = 73
    dp = x_1 - x_0
    alpha = -((x_1 + x_0) / 2 - 768 / 2) / (768 / 2) * fov / 2 / 180 * np.pi * 1.08
    alpha = tiefpass(alpha, alpha_old, 0.8)
    distance = de * f / dp * (1 + abs(alpha) / 4 * 1.1) * (1 - abs(drehung / 100) / 2)
    distance = np.sqrt(distance ** 2 - hoehe_cam ** 2)

    x = np.cos(alpha) * distance
    y = np.sin(alpha) * distance
    print("x,y",x,y)



    accell_in,steering=controller_verfolgen(x,y,alpha)
    print("accel",accell_in,steering)
    # cv2.imshow("Image Window", frame)
    # cv2.waitKey(1)
    #
    # circle = cv2.circle(frame, (x_1, y_1), 5, 120, -1)
    # circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
    # cv2.imshow("Image Window", circle)
    # cv2.waitKey(1)

    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = accell_in  # aktuell in tick rate(+- 3900)
    message.point.y = 2  # not used
    message.point.z = steering  # in grad(max +-20)
    # rospy.loginfo(message)
    pub.publish(message)
    rate.sleep()
    # cv2.waitKey()


def listener():
    # tracker = tracking_red_dots(308,410)
    # tracker = tracking_red_dots(960, 1280,350,900,400,960)
    tracker = tracking_red_dots(576, 768)

    rospy.Subscriber("/raspicam_node/image", Image, callback, tracker)
    rospy.spin()
    # video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
