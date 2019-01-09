
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image,CompressedImage
from tensorflow import keras
import cv2
import rospy
from os import listdir
from os.path import isfile, join
import numpy as np

def resize_image(frame):#also crops image
    tmp = frame[1:].reshape(308, 410)
    tmp=tmp[int(0.2*308.0):308, 0:410]

    rescaled = cv2.resize(tmp, (64, 36))
    # print(rescaled.shape)

    return np.hstack([frame[0], rescaled.flatten()])

validation_data = np.array(np.loadtxt(open('/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/validation_data/validation_data_full.csv',"rb"),delimiter=","))
img_rows, img_cols = 36, 64
input_shape = (img_rows, img_cols, 1)

x_val = np.ones((validation_data.shape[0], img_rows, img_cols, 1))
#y_val = np.zeros(validation_data.shape[0])
print("val loaded")
for i in range(0, validation_data.shape[0]):
    x_val[i, :, :] = resize_image(validation_data[i,])[1:].reshape(1, img_rows, img_cols, 1)/255
y_val=validation_data[:,0]/29
#print(x_val/255)
#print(y_val/29.)


mypath="/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/modelle_week_random"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print("validation_done")
validation_list=list()
for file_name in onlyfiles:
    print(file_name)
    model = keras.models.load_model(mypath +"/" + file_name)

    validation_list.append((model.evaluate(x_val, y_val),file_name))
print(validation_list)