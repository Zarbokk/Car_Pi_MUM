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
    #print("1")
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    #print("2")
    StartTime = time.time()
    StopTime = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
        if time.time()-StopTime>0.01:
            print("waiting_for_signal_done")
            break

    #print("3")
    while GPIO.input(GPIO_ECHO) == 1:
	    StopTime = time.time()
    #print("4")
    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    #print("5")
    return distance






def talker():
    old_distance=0
    while not rospy.is_shutdown():
        d = distance()
        if d>350 or d<5:
            d=old_distance
        d=d*0.9+old_distance*0.1
        message = Range()
        message.header.stamp = rospy.Time.now()
        message.range = d  # very noisy
        #rospy.loginfo(message)
        pub.publish(message)
        rate.sleep()
        old_distance = d
    GPIO.cleanup()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
