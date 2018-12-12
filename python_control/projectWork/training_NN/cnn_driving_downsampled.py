import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from AirSimClient import *

constant_speed=5


def maxValue(regulate,max_value):
    if regulate>max_value:
        regulate=max_value
    if regulate<-max_value:
        regulate=-max_value
    return regulate
def getImage(client):
    responses = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

    # reshape array to 4 channel image array H X W X 4
    mg_rgba = img1d.reshape(response.height, response.width, 4)
    grey_image = cv2.cvtColor(mg_rgba, cv2.COLOR_BGR2GRAY)
    return grey_image
img_rows, img_cols = 36, 64
input_shape = (img_rows, img_cols, 1)
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
model.load_weights("E:/ML_MODELS/model_for_cicle/cicle_model")

client = CarClient()
car_controls = CarControls()
client.confirmConnection()
client.enableApiControl(True)
client.reset()
print("startWhile")
while True:
    grey_image=getImage(client)

    cv2.imshow('image', grey_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    grey_image = cv2.resize(grey_image.reshape(144, 256), (64, 36))
    grey_image=np.array(grey_image)/255
    grey_image = grey_image.reshape(1, img_rows, img_cols, 1)
    steering = model.predict(grey_image)[0][0]

    car_state = client.getCarState()
    accell_in = 10 * (constant_speed - car_state.speed)
    accell_in = maxValue(accell_in, 1)
    print(steering)
    car_controls.steering = steering.astype(float)
    car_controls.throttle = accell_in
    car_controls.is_manual_gear = False;
    if (accell_in < 0):
        car_controls.is_manual_gear = True;
        car_controls.manual_gear = -1
    client.setCarControls(car_controls)


client.enableApiControl(False)