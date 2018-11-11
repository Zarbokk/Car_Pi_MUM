#!/usr/bin/env python
# license removed for brevity
import time
import rospy
from geometry_msgs.msg import Pose2D
def talker():
    pub = rospy.Publisher('car_motor_input', Pose2D, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    input_motor_speed=0
    while not rospy.is_shutdown():
        num = raw_input('Geschwindigkeit:')
        try:
            input_motor_speed=float(num)
        except ValueError:
            print('should have used a number')
            continue

        message = Pose2D()
        message.x=input_motor_speed#aktuell in tick rate(+- 3900)
        message.y=2#not used
        message.theta=0#in grad(max +-20)
        rospy.loginfo(message)
        pub.publish(message)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
