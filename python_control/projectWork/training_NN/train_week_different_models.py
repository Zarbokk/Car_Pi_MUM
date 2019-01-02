import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import cv2

steps_per_epoche = 4158
number_epochs = 5

n_kernels = (16, 32)
kernel_size = (3, 4, 5)
number_fc = (1, 2)
number_neurons = (64, 128)
learning_rates = (0.001, 0.002, 0.0005)
batch_size = 128


def generate_arrays_from_file(path, printing):
    while 1:
        f = open(path)
        i = 0
        img_rows, img_cols = 36, 64
        x_train = np.zeros((batch_size, img_rows, img_cols, 1))
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

            x_train[i, :, :] = x_train_tmp.reshape(1, img_rows, img_cols, 1)
            y_train[i] = y_train_tmp
            if printing:
                print(y_train)
            # print(x_train.shape,y_train.shape)
            i = i + 1

            if i == batch_size:
                print(i, x_train.shape)
                x_train = x_train.astype('float32')
                y_train = y_train.astype('float32')
                yield (x_train, y_train)
                i = 0

        f.close()


img_rows, img_cols = 36, 64
input_shape = (img_rows, img_cols, 1)
# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, 1)
# y_test = keras.utils.to_categorical(y_test, 1)
# time.sleep(10)

for i in range(0, 2):
    for j in range(0, 3):
        for k in range(0, 2):
            for l in range(0, 2):
                for m in range(0, 3):

                    model = keras.Sequential()
                    model.add(keras.layers.Conv2D(n_kernels[i], kernel_size=(kernel_size[j], kernel_size[j]),
                                                  activation='relu', input_shape=input_shape))
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                    model.add(
                        keras.layers.Conv2D(n_kernels[i] * 2, (kernel_size[j], kernel_size[j]), activation='relu'))
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                    model.add(keras.layers.Flatten())
                    model.add(keras.layers.Dense(number_neurons[l], activation='relu'))
                    model.add(keras.layers.Dropout(0.5))
                    if number_fc[k] == 2:
                        model.add(keras.layers.Dense(number_neurons[l], activation='relu'))
                        model.add(keras.layers.Dropout(0.5))
                    model.add(keras.layers.Dense(1, activation='linear'))

                    model.compile(loss=keras.losses.mean_squared_error,
                                  optimizer=keras.optimizers.Adam(lr=learning_rates[m]))

                    model.fit_generator(generate_arrays_from_file(
                        "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/complete_for_training/augmented_data_FULL.csv",
                        False),
                        steps_per_epoch=steps_per_epoche, epochs=number_epochs)
                    model.save(
                        "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/week_models_20_{0}_{1}_{2}_{3}_{4}.h5".format(
                            i, j, k, l, m))
                    print(i, j, k, l, m,)
