#!/usr/bin/env python
# license removed for brevity
import time
import rospy
import invModelControlROS as imcr
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped

pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=1)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(100)  # Frequenz der Anwendung
start_time = 0
angle = 0


def talker(Imu_data, inverse_model):

    global angle
    global start_time

    angle = angle + Imu_data.angular_velocity.z
    real_angle=angle

    endtime = inverse_model.trajectory.specifics[0]
    print(start_time)

    if start_time < 1:
        v = 1000
        delta = 0
    else:
        v, delta = inverse_model.carInput(start_time - 1)
    if start_time > endtime + 0.5:
        v = 0
        delta = 0
    v=0
    delta=0
    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = v / 2.2 * 4095  # aktuell in tick rate(+- 3900)
    message.point.y = real_angle*90/1000000  # not used
    message.point.z = delta * 180 / 3.14159  # in grad(max +-20)
    rospy.loginfo(message)
    pub.publish(message)
    rate.sleep()
    start_time = start_time + 0.01


def subscribe():
    inverse_model = imcr.invModelControl(1.1 / 2, 0.5, "qubicS")

    while not rospy.is_shutdown():
        rospy.Subscriber('IMU_acceleration', Imu, talker, inverse_model)
        rospy.spin()

#+90
if __name__ == '__main__':
    try:
        subscribe()
    except rospy.ROSInterruptException:
        pass
