#!/usr/bin/env python
# license removed for brevity
import time
import rospy
import invModelControlforROS as imcr

from geometry_msgs.msg import PointStamped


def talker():
    pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(100)  # Frequenz der Anwendung
    input_motor_speed = 0

    start_time = 0
    inverse_model = imcr.invModelControlROS(0.5, 1.1)
    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x =2000  # aktuell in tick rate(+- 3900)
    message.point.y = 2  # not used
    message.point.z = 0  # in grad(max +-20)
    rospy.loginfo(message)
    pub.publish(message)
    time.sleep(1)
    while not rospy.is_shutdown():

        print(start_time)
        u = inverse_model.carInput(start_time)
        v = u[0]
        delta = u[1]
        if start_time>2:
            v=0
        message = PointStamped()
        message.header.stamp = rospy.Time.now()
        message.point.x = v / 2.2 * 4095  # aktuell in tick rate(+- 3900)
        message.point.y = 2  # not used
        message.point.z = delta * 180 / 3.14159  # in grad(max +-20)
        rospy.loginfo(message)
        pub.publish(message)
        rate.sleep()
        input_motor_speed=delta
        start_time = start_time + 0.01


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
