# !/usr/bin/env python
from red_dots_tracking_class import tracking_red_dots
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
import cv2
import rospy
import numpy as np
import io
import time
from picamera.array import PiRGBArray
from picamera import PiCamera



#x_0_true = y_0_true = x_1_true = y_1_true = 0
#cv2.CV_8UC3

def talker():
    first_run = True
    #x_0_true, y_0_true, x_1_true, y_1_true = 0
    rospy.init_node('publisher', anonymous=True)
    pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=0)
    rate = rospy.Rate(20)  # Frequenz der Anwendung
    #tracker = tracking_red_dots(960, 1280, 350, 900, 400, 960)
    tracker = tracking_red_dots(576, 768, 210, 570, 240, 576)
    # tracker = tracking_red_dots(308, 410, 0, 410, 0, 308)
    # tracker = tracking_red_dots(960, 1280, 0, 1280, 0, 960)


    camera = PiCamera()
    camera.resolution = (768,576)
    camera.framerate = 20
    rawCapture = PiRGBArray(camera, size=(768, 576))

    time.sleep(0.1)
    camera.sharpness = 0
    camera.contrast = 50
    camera.brightness = 80
    camera.saturation = 100
    camera.ISO = 100
    camera.video_stabilization = False
    camera.exposure_compensation = 0
    camera.exposure_mode = 'auto'
    camera.meter_mode = 'average'
    camera.awb_mode = 'auto'
    camera.image_effect = 'none'
    camera.color_effects = None
    camera.rotation = 0
    camera.hflip = False
    camera.vflip = False
    camera.crop = (0.0, 0.0, 1.0, 1.0)
    # allow the camera to warmup
    time.sleep(0.1)

    # grab an image from the camera

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # YOURS:  for frame in camera.capture_continuous(stream, format="bgr",  use_video_port=True):
            # Truncate the stream to the current position (in case
            # prior iterations output a longer image)
#cv2.COLORBGR
        #camera.capture(rawCapture, format="bgr")
        frame =np.asarray(frame.array,dtype=np.uint8) #rawCapture.array
        if first_run:
            x_0_true, y_0_true, x_1_true, y_1_true = tracker.get_red_pos(frame)
            first_run = False
        else:
            x_0, y_0, x_1, y_1 = tracker.get_red_pos(frame)
            #cv2.imshow('largest contour', frame)
            #cv2.waitKey(1)

            #circle = cv2.circle(frame, (x_1, y_1), 5, 120, -1)
            #circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
            #cv2.imshow("Image Window", circle)
            #cv2.waitKey(1)
            # print("distance from point start:",np.sqrt((x_0_true-x_1_true)*(x_0_true-x_1_true)+(y_0_true-y_1_true)*(y_0_true-y_1_true)))
            # print("distance current:",
            #      np.sqrt((x_0 - x_1) * (x_0 - x_1) + (y_0 - y_1) * (y_0 - y_1)))

            # print("x Distance start",x_0_true-x_1_true)
            # print("x Distance current", x_0 - x_1)
            # print("y Distance start", y_0_true - y_1_true)
            # print("y Distance current", y_0 - y_1)
            # cv2.destroyAllWindows()
            # video.release()
            # (rows, cols, channels) = cv_image.shape
            # if cols > 60 and rows > 60:
            #     cv2.circle(cv_image, (50, 50), 10, 255)
            # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            distance = np.sqrt((x_0_true - x_1_true) * (x_0_true - x_1_true) + (y_0_true - y_1_true) * (
                        y_0_true - y_1_true)) - np.sqrt((x_0 - x_1) * (x_0 - x_1) + (y_0 - y_1) * (y_0 - y_1))
            # print("distance:", distance)

            accel_in = feedbackControler(distance)
            #accel_in = feedbackControler(0)

            message = PointStamped()
            message.header.stamp = rospy.Time.now()
            message.point.x = accel_in  # aktuell in tick rate(+- 3900)
            message.point.y = 2  # not used
            message.point.z = 0  # in grad(max +-20)
            rospy.loginfo(message)
            pub.publish(message)
            rate.sleep()
            #if 0xFF == ord('q'):
            #    break
        rawCapture.truncate(0)
        # reset the stream before the next capture

    #rospy.spin()
    #cv2.destroyAllWindows()
    #cv2.waitKey()

def feedbackControler(error):
    MAXERR=60
    MAXU=4095
    #if error<=0:
    #    return 0
    #else:
    error=error/MAXERR
    gain=0.3
    return gain*error*MAXU


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # 1280
    # x960
    #tracker = tracking_red_dots(308,410)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video = cv2.VideoWriter('/home/tim/Dokumente/Video_car_find.avi', fourcc, 20.0, (1280, 960))

    #video=3
    #rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/raspicam_node/image", Image, callback, tracker)



if __name__ == '__main__':
    talker()
