# !/usr/bin/env python
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image,CompressedImage
from tensorflow import keras
import cv2
import rospy
import numpy as np
#update

img_rows, img_cols = 36, 64
input_shape = (img_rows, img_cols, 1)
graph = tf.get_default_graph()
model= keras.models.load_model(
    "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/test_model_complete_64x36.h5")
rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('car_motor_input', PointStamped, queue_size=0)
rate = rospy.Rate(25)  # Frequenz der Anwendung



def callback(image):
    global graph,model
    #print(image.encoding)
    brige = CvBridge()
    try:
        #brige.
        #frame=brige.imgmsg_to_cv2(image, "bgr8")
        #frame = brige.imgmsg_to_cv2(image, "passthrough")
        frame = brige.compressed_imgmsg_to_cv2(image, "passthrough")
    except CvBridgeError as e:
        print(e)
    cv2.imshow("Image Window", frame)
    cv2.waitKey(1)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rescaled = cv2.resize(gray_image, (64, 36))

    rescaled=rescaled.astype('float32')/255
    print(rescaled)
    print(rescaled.reshape(1, 36, 64, 1).shape)
    with graph.as_default():
        #print(model.predict((rescaled.reshape(1, 36, 64, 1)).astype('float32')))
        steering_in=model.predict(rescaled.reshape(1, 36, 64, 1))[0][0]
        #print(steering_in)
        #print(steering_in*29)
        steering_in=steering_in*29
    if steering_in>29:
        steering_in=29
    if steering_in<-29:
        steering_in=-29

    message = PointStamped()
    message.header.stamp = rospy.Time.now()
    message.point.x = 1000  # aktuell in tick rate(+- 3900)
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

    #video=3
    #rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback)
    rospy.spin()
    #video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
