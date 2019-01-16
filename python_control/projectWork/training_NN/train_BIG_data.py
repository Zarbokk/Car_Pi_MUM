import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import cv2

steps_per_epoche=10
number_epochs=10

kernel_size_first = 3
conv_first = 16
kernel_size_second = 2
conv_second = 32
kernel_size_third = 3
conv_third = 64
kernel_size_fourth = 3
conv_fourth = 128


learning_rate=0.001
batch_size = 64

def generate_arrays_from_file(path, printing):
    while 1:
        f = open(path)
        i = 0
        img_rows, img_cols = 36, 64
        x_train = np.zeros((batch_size, img_rows, img_cols,1))
        y_train = np.zeros(batch_size)
        for line in f:
            myarray = np.fromstring(line, dtype=float, sep=',')
            x_train_tmp = np.asarray(myarray[1:])
            # print(x_train)
            y_train_tmp = np.asarray([myarray[0]])
            # print(y_train)
            # input image dimensions

            # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
            # the data, split between train and test sets
            # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            x_train[i, :, :] = x_train_tmp.reshape(1, img_rows, img_cols,1)
            y_train[i] = y_train_tmp
            if printing:
                print(y_train)
            # print(x_train.shape,y_train.shape)
            i = i + 1

            if i == batch_size:
                print(i,x_train.shape)
                x_train = x_train.astype('float32')
                y_train = y_train.astype('float32')
                yield (x_train, y_train)
                i = 0

        f.close()


img_rows, img_cols = 36, 64
input_shape = (img_rows, img_cols,1)
# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, 1)
# y_test = keras.utils.to_categorical(y_test, 1)
# time.sleep(10)
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=learning_rate))

model.fit_generator(generate_arrays_from_file(
    "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/complete_for_training/augmented_data_FULL.csv",
    False),
                    steps_per_epoch=steps_per_epoche, epochs=number_epochs)
# model.fit_generator(generate_arrays_from_file("F:/OneDrive/Uni/StudienArbeit/Car_Dataset/augmented_data_FULL.csv",),
#                    steps_per_epoch=10000, epochs=100)

# model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          shuffle=True,
#          validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score)
#print(model.predict_generator(generate_arrays_from_file(
#    "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/complete_for_training/augmented_data_FULL.csv",
#    True), 10, max_queue_size=1))
# print(y_train)

#model.save("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/test_model_complete_64x36.h5")
# model.save_weights("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/test_model_64x36.h5")
