# !/usr/bin/env python
"""Script to listen for car_motor_input topic.

This has to run on the car directly and translates the communicated speed and
angle to the correct board commands.
"""

import rospy
from geometry_msgs.msg import PointStamped
import Adafruit_PCA9685

motor_input_value = 0
angle = 0


def callback(data):
    """Callback to pass speed and angle to `send_data`."""
    global motor_input_value
    motor_input_value = data.point.x
    global angle
    angle = data.point.z


def send_data(pwm):
    """Send angle and speed data to motor."""
    if angle < 30 and angle > -30:
        pwm.set_pwm(6, 0, 2407 + 869 * int(angle) / 30)
        # 2457 da auf 400 Hz 30 fuer 30 Grad(in de$
    if motor_input_value >= -4095 and motor_input_value <= 0:
        # moving backwards
        pwm.set_pwm(11, 0, 0)
        pwm.set_pwm(10, 0, -1 * int(motor_input_value))
    if motor_input_value <= 4095 and motor_input_value >= 0:
        # moving forward
        pwm.set_pwm(11, 0, int(motor_input_value))
        pwm.set_pwm(10, 0, 0)


def listener():
    """Motor Input Listener.

    In ROS, nodes are uniquely named. If two nodes with the same
    node are launched, the previous one is kicked off. The
    anonymous=True flag means that rospy will choose a unique
    name for our 'listener' node so that multiple listeners can
    run simultaneously.
    """
    pwm = Adafruit_PCA9685.PCA9685(address=0x40)
    pwm.set_pwm_freq(400)  # Set frequency to 400 Hz
    pwm.set_pwm(8, 0, 4000)
    pwm.set_pwm(9, 0, 4000)
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("car_motor_input", PointStamped, callback)
    rate = rospy.Rate(30)
    # spin() simply keeps python from exiting until this node is stopped
    while not rospy.is_shutdown():
        rate.sleep()
        send_data(pwm)
        # rospy.spin()

if __name__ == '__main__':
    listener()
