#!/usr/bin/env python
# license removed for brevity

import smbus
import math
import rospy
from std_msgs.msg import String
power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c
def read_byte(reg):
    return bus.read_byte_data(address, reg)
def read_word(reg):
    h = bus.read_byte_data(address, reg)
    l = bus.read_byte_data(address, reg + 1)
    value = (h << 8) + l
    return value
def read_word_2c(reg):
    val = read_word(reg)
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
    pub = rospy.Publisher('IMU_acceleration', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time609264()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
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



















#!/usr/bin/python


# Register





bus = smbus.SMBus(1)  # bus = smbus.SMBus(0) fuer Revision 1
address = 0x68  # via i2cdetect

# Aktivieren, um das Modul ansprechen zu koennen
bus.write_byte_data(address, power_mgmt_1, 0)

print "Gyroskop"
print "--------"

gyroskop_xout = read_word_2c(0x43)
gyroskop_yout = read_word_2c(0x45)
gyroskop_zout = read_word_2c(0x47)

print
"gyroskop_xout: ", ("%5d" % gyroskop_xout), " skaliert: ", (gyroskop_xout / 131)
print
"gyroskop_yout: ", ("%5d" % gyroskop_yout), " skaliert: ", (gyroskop_yout / 131)
print
"gyroskop_zout: ", ("%5d" % gyroskop_zout), " skaliert: ", (gyroskop_zout / 131)

print
print
"Beschleunigungssensor"
print
"---------------------"

beschleunigung_xout = read_word_2c(0x3b)
beschleunigung_yout = read_word_2c(0x3d)
beschleunigung_zout = read_word_2c(0x3f)

beschleunigung_xout_skaliert = beschleunigung_xout / 16384.0
beschleunigung_yout_skaliert = beschleunigung_yout / 16384.0
beschleunigung_zout_skaliert = beschleunigung_zout / 16384.0

print
"beschleunigung_xout: ", ("%6d" % beschleunigung_xout), " skaliert: ", beschleunigung_xout_skaliert
print
"beschleunigung_yout: ", ("%6d" % beschleunigung_yout), " skaliert: ", beschleunigung_yout_skaliert
print
"beschleunigung_zout: ", ("%6d" % beschleunigung_zout), " skaliert: ", beschleunigung_zout_skaliert

print
"X Rotation: ", get_x_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert)
print
"Y Rotation: ", get_y_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert)

