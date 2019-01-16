# !/usr/bin/env python
#from red_dots_tracking_class import tracking_red_dots
#from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Range
#import cv2
import rospy
#import numpy as np

import RPi.GPIO as GPIO
import time




rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('distance_sensor_03', Range, queue_size=1)
rate = rospy.Rate(100)  # Frequenz der Anwendung

GPIO.setmode(GPIO.BCM)

GPIO_TRIGGER = 4
GPIO_ECHO = 13

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
	StartTime = time.time()

    while GPIO.input(GPIO_ECHO) == 1:
	StopTime = time.time()

    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2

    return distance






def talker():
    old_distance=0
    while not rospy.is_shutdown():
        d = distance()
        if d>350:
            d=old_distance
        d=d*0.8+old_distance*0.2
        message = Range()
        message.header.stamp = rospy.Time.now()
        message.range = d  # very noisy
        rospy.loginfo(message)
        pub.publish(message)
        rate.sleep()
        old_distance = d


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
