# !/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped
import Adafruit_PCA9685

def callback(data, pwm):
    if data.point.z<30 and data.point.z>-30:
        pwm.set_pwm(1,0,2457+819*int(data.point.z)/30)#2457 da auf 400 Hz 30 fuer 30 Grad(in der theorie)
    if data.point.x>=-4095 and data.point.x<=0:
        pwm.set_pwm(11,0,0)
        pwm.set_pwm(10, 0, int(data.point.x)*-1)
    if data.point.x<=4095 and data.point.x>=0:
        pwm.set_pwm(11,0,int(data.point.x))
        pwm.set_pwm(10, 0, 0)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    pwm = Adafruit_PCA9685.PCA9685(address=0x40)
    pwm.set_pwm_freq(400)#frequenz von 400 Hz
    pwm.set_pwm(8,0,4000)
    pwm.set_pwm(9,0,4000)
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("car_motor_input", PointStamped, callback, pwm)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()