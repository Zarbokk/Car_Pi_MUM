# !/usr/bin/env python
from red_dots_tracking_class import tracking_red_dots
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CompressedImage#Image
import cv2
import rospy
import numpy as np


rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=0)
rate = rospy.Rate(25)  # Frequenz der Anwendung

first_run = True

x_0_true = y_0_true = x_1_true = y_1_true = 0
#cv2.CV_8UC3

car_input_last_step=0
steering_angle_last_step=0



def distanceControl(error,steering_in):
    global car_input_last_step, steering_angle_last_step
    MAXERR = 60
    MAXU = 4095
    MINU = 1000
    MAXSTEERING = 29
    TOL = 0.1
    error = error / MAXERR
    if error <= TOL:
        return 0
    gain = 0.7
    k = 6
    u = gain*error*MAXU*(1-np.abs(steering_in)/(MAXSTEERING*k))

    return np.max([u, MINU])


def angleControl(Point_left, Point_right, u0):
    global car_input_last_step, steering_angle_last_step
    '''
    PARAMETERS
    ----------
    Point_left : nparray 2-Dim
        Point_left[0] x
        Point_left[1] y

    Point_right : nparray 2-Dim
        Point_right[0] x
        Point_right[1] y

    u0 : double
        u0 midline of intrinsic matrix
    '''
    Point_left = np.asarray(Point_left)
    Point_right = np.asarray(Point_right)
    Threshold = u0 / 4
    MAXSTEERING = 29
    Midpoint = (Point_right + Point_left) / 2
    HorizontalError = (u0 - Midpoint[0]) / Threshold
    if np.abs(HorizontalError) < 1:
        k = 1
        # possible other control schemes
        return -k * HorizontalError * MAXSTEERING
    else:
        return -np.sign(HorizontalError) * MAXSTEERING

def callback(image,tracker):
    #print(image.encoding)
    brige = CvBridge()
    try:
        #brige.
        #frame=brige.imgmsg_to_cv2(image, "bgr8")
        #frame = brige.imgmsg_to_cv2(image, "passthrough")
        frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    global first_run
    global x_0_true, y_0_true, x_1_true, y_1_true
    if first_run:
        #cv2.imshow("Image Window", frame)
        #cv2.waitKey(1)
        x_0_true, y_0_true, x_1_true, y_1_true = tracker.get_red_pos(frame)
        first_run = False
    else:




        x_0, y_0, x_1, y_1 = tracker.get_red_pos(frame)
        #frame = frame[250:576, 156:612]
        #cv2.imwrite("/home/tim/Dokumente/poster/crob_image.png", frame)
        cv2.imshow("Image Window", frame)
        cv2.waitKey(1)
        #print(x_0,y_0,x_1,y_1)
        f = 592.61
        de = 70
        dp = x_1 - x_0
        print("distance to car:",de * f / dp)
        #circle = cv2.circle(frame, (x_1, y_1), 5, 120, -1)
        #circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
        #cv2.imshow("Image Window", circle)
        #cv2.waitKey(1)
        #print("distance from point start:",np.sqrt((x_0_true-x_1_true)*(x_0_true-x_1_true)+(y_0_true-y_1_true)*(y_0_true-y_1_true)))
        #print("distance current:",
        #      np.sqrt((x_0 - x_1) * (x_0 - x_1) + (y_0 - y_1) * (y_0 - y_1)))

        #print("x Distance start",x_0_true-x_1_true)
        #print("x Distance current", x_0 - x_1)
        #print("y Distance start", y_0_true - y_1_true)
        #print("y Distance current", y_0 - y_1)
        #cv2.destroyAllWindows()
        #video.release()
        # (rows, cols, channels) = cv_image.shape
        # if cols > 60 and rows > 60:
        #     cv2.circle(cv_image, (50, 50), 10, 255)
        #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)



        distance=np.sqrt((x_0_true-x_1_true)*(x_0_true-x_1_true)+(y_0_true-y_1_true)*(y_0_true-y_1_true))-np.sqrt((x_0 - x_1) * (x_0 - x_1) + (y_0 - y_1) * (y_0 - y_1))
        #print("distance:", distance)

        steering_in = angleControl([x_0, y_0], [x_1, y_1], 768 / 2)
        accel_in = distanceControl(distance, steering_in)



        global car_input_last_step,steering_angle_last_step


        car_input_last_step = accel_in
        steering_angle_last_step=steering_in
        #accel_in=0

        message = PointStamped()
        message.header.stamp = rospy.Time.now()
        message.point.x = accel_in  # aktuell in tick rate(+- 3900)
        message.point.y = 2  # not used
        message.point.z = steering_in  # in grad(max +-20)
        rospy.loginfo(message)
        pub.publish(message)
        rate.sleep()
    #cv2.waitKey()



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
    #tracker = tracking_red_dots(960, 1280,350,900,400,960)
    tracker = tracking_red_dots(576, 768, 156, 612, 250, 576)
    #tracker = tracking_red_dots(308, 410, 0, 410, 0, 308)
    #tracker = tracking_red_dots(960, 1280, 0, 1280, 0, 960)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video = cv2.VideoWriter('/home/tim/Dokumente/Video_car_find.avi', fourcc, 20.0, (1280, 960))

    #video=3
    #rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback, tracker)
    rospy.spin()
    #video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
