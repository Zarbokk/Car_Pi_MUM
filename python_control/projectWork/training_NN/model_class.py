#! /usr/bin/env python
"""Class for Tracking via recognizing red shock absorbers."""
import cv2
import numpy as np
from tensorflow import keras

class keras_model:

    def __init__(self):
        img_rows, img_cols = 36, 64
        input_shape = (img_rows, img_cols, 1)
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        self.model.load_weights(
            "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/first_model_64x36.h5")

    def get_predicted_steering(self, frame):
        print(frame.shape)
        self.model.predict(frame.reshape(1, 36, 64, 1))

        steering=0
        return steering