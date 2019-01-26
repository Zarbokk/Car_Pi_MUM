#!/usr/bin/env python
# license removed for brevity

import smbus
import math
import rospy
import tf.transformations
from sensor_msgs.msg import Imu


def read_byte(reg,bus,address):
    return bus.read_byte_data(address, reg)
def read_word(reg,bus,address):
    h = bus.read_byte_data(address, reg)
    l = bus.read_byte_data(address, reg + 1)
    value = (h << 8) + l
    return value
def read_word_2c(reg,bus,address):
    val = read_word(reg,bus,address)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val
def dist(a, b):
    return math.sqrt((a * a) + (b * b))
def get_y_rotation(x, y, z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)
def get_x_rotation(x, y, z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)
def talker():
    pub = rospy.Publisher('IMU_03', Imu, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(50) # 10hz

    power_mgmt_1 = 0x6b
    power_mgmt_2 = 0x6c
    bus = smbus.SMBus(1)  # bus = smbus.SMBus(0) fuer Revision 1
    address = 0x68  # via i2cdetect

    # Aktivieren, um das Modul ansprechen zu koennen
    bus.write_byte_data(address, power_mgmt_1, 0)

    while not rospy.is_shutdown():
        imu_data = Imu()
        gyroskop_xout = read_word_2c(0x43,bus,address)
        gyroskop_yout = read_word_2c(0x45,bus,address)
        gyroskop_zout = read_word_2c(0x47,bus,address)
        beschleunigung_xout = read_word_2c(0x3b,bus,address)
        beschleunigung_yout = read_word_2c(0x3d,bus,address)
        beschleunigung_zout = read_word_2c(0x3f,bus,address)
        beschleunigung_xout_skaliert = beschleunigung_xout / 16384.0
        beschleunigung_yout_skaliert = beschleunigung_yout / 16384.0
        beschleunigung_zout_skaliert = beschleunigung_zout / 16384.0

        imu_data.linear_acceleration.x = beschleunigung_xout
        imu_data.linear_acceleration.y = beschleunigung_yout
        imu_data.linear_acceleration.z = beschleunigung_zout
        imu_data.angular_velocity.x = gyroskop_xout
        imu_data.angular_velocity.y = gyroskop_yout
        imu_data.angular_velocity.z = gyroskop_zout
        quaternion = tf.transformations.quaternion_from_euler(
            get_x_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert),
            get_y_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert), 0)
        imu_data.orientation.x = quaternion[0]
        imu_data.orientation.y = quaternion[1]
        imu_data.orientation.z = quaternion[2]
        imu_data.orientation.w = quaternion[3]
        imu_data.header.stamp = rospy.Time.now()
        #rospy.loginfo(imu_data)
        pub.publish(imu_data)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass