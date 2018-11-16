#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file presents an interface for interacting with the Playstation 4 Controller
# in Python. Simply plug your PS4 controller into your computer using USB and run this
# script!
#
# NOTE: I assume in this script that the only joystick plugged in is the PS4 controller.
#       if this is not the case, you will need to change the class accordingly.
#
# Copyright © 2015 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import pprint
import pygame





#!/usr/bin/env python
# license removed for brevity
import time
import rospy
from geometry_msgs.msg import PointStamped
def talker():
    controller = None
    axis_data = None
    button_data = None
    hat_data = None
    pygame.init()
    pygame.joystick.init()
    controller = pygame.joystick.Joystick(0)
    controller.init()
    if not axis_data:
        axis_data = {}

    if not button_data:
        button_data = {}
        for i in range(controller.get_numbuttons()):
            button_data[i] = False

    if not hat_data:
        hat_data = {}
        for i in range(controller.get_numhats()):
            hat_data[i] = (0, 0)


    pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(30) # 10hz
    input_motor_speed=0
    while not rospy.is_shutdown():
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_data[event.axis] = round(event.value, 2)
            elif event.type == pygame.JOYBUTTONDOWN:
                button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                hat_data[event.hat] = event.value

            # Insert your code on what you would like to happen for each event here!
            # In the current setup, I have the state simply printing out to the screen.

            os.system('clear')
        num=axis_data[1]*-2000
        angle=axis_data[0]*25

        input_motor_speed=float(num)
        message = PointStamped()
        message.header.stamp = rospy.Time.now()
        #message.header.frame_id =1
        #message.header.seq =2
        message.point.x=input_motor_speed#aktuell in tick rate(+- 3900)
        message.point.y=2#not used
        message.point.z=float(angle)#in grad(max +-20)
        rospy.loginfo(message)
        pub.publish(message)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass