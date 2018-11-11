# !/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose2D
import Adafruit_PCA9685

def callback(data, pwm):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.x)
    if data.theta<30 and data.theta>-30:
        pwm.set_pwm(1,0,2457+819*int(data.theta)/30)
    if data.x>-3900 and data.x<=0:
        pwm.set_pwm(11,0,0)
        pwm.set_pwm(10, 0, int(data.x)*-1)
    if data.x<3900 and data.x>=0:
        pwm.set_pwm(11,0,int(data.x))
        pwm.set_pwm(10, 0, 0)


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    pwm = Adafruit_PCA9685.PCA9685(address=0x40)
    pwm.set_pwm_freq(400)
    pwm.set_pwm(8,0,3500)
    pwm.set_pwm(9,0,3500)
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("chatter", Pose2D, callback, 3)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


    if __name__ == '__main__':
        listener()