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
geschwindigkeit=1.1*1.5

def talker(Imu_data, inverse_model):
    global angle
    global start_time
    y = 0
    error=0
    psi=0
    scaling_imu_angle = 17063036.0 / 4.0 / 360.0
    angle = angle + Imu_data.angular_velocity.z
    real_angle = angle

    endtime = inverse_model.trajectory.specifics[0]
    print(start_time, endtime)

    if start_time < 1:
        v = geschwindigkeit
        delta = 0
    else:
        v, delta, psi, y = inverse_model.carInput(start_time - 1)
        psi = -psi * 180 / 3.14159
        delta=-delta
        print (psi, real_angle / scaling_imu_angle)
        error = psi-real_angle / scaling_imu_angle
        inverse_correction=inverse_model.trajectoryControler(error, 1.5*20)/180*3.14159
        inverse_correction=error*2/180.0*3.14159
        delta = delta + inverse_correction
        print(delta,inverse_correction)

    if start_time > 2:
        v = 0
        #delta = 0
    # v = 0
    # delta = 0
    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = v / 2.2 * 4095  # aktuell in tick rate(+- 3900)
    message.point.y = psi  # not used
    message.point.z = delta * 180 / 3.14159  # in grad(max +-20)
    rospy.loginfo(message)
    pub.publish(message)
    rate.sleep()
    start_time = start_time + 0.01


def subscribe():
    inverse_model = imcr.invModelControl(geschwindigkeit, 0.7, "sShape")

    while not rospy.is_shutdown():
        rospy.Subscriber('IMU_acceleration', Imu, talker, inverse_model)
        rospy.spin()


# +90
if __name__ == '__main__':
    try:
        subscribe()
    except rospy.ROSInterruptException:
        pass
