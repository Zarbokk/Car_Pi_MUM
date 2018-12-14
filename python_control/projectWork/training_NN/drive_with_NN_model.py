# !/usr/bin/env python
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image,CompressedImage
from tensorflow import keras
import cv2
import rospy
import numpy as np


img_rows, img_cols = 36, 64
input_shape = (img_rows, img_cols, 1)
graph = tf.get_default_graph()
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='linear'))
model.load_weights("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/first_model_64x36.h5")

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
        print(model.predict((rescaled.reshape(1, 36, 64, 1)).astype('float32')))
        steering_in=model.predict(rescaled.reshape(1, 36, 64, 1))[0][0]
        print(steering_in)
        print(steering_in*29)
        steering_in=steering_in*29*1.5
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
    train_data = np.array(np.loadtxt(
        open(
            "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/augmented_grey_image_data_0.csv",
            "rb"),
        delimiter=",", skiprows=1))
    img_rows, img_cols = 36, 64
    x_train = train_data[0:train_data.shape[0], 1:]
    y_train = train_data[0:train_data.shape[0], 0]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')

    with graph.as_default():
        steering_in=model.predict(x_train.reshape(x_train.shape[0], 36, 64, 1))
        print(steering_in*29)
        print(y_train*29)
    #video=3
    #rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, callback)
    rospy.spin()
    #video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
