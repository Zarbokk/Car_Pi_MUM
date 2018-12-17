import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import cv2
batch_size = 32
epochs = 20

def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            myarray = np.fromstring(line, dtype=float, sep=',')
            x_train = np.asarray(myarray[1:])
            #print(x_train)
            y_train = np.asarray([myarray[0]])
            #print(y_train)
            # input image dimensions
            img_rows, img_cols = 36, 64
            #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
            # the data, split between train and test sets
            # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            x_train = x_train.reshape(1, img_rows, img_cols, 1)


            x_train = x_train.astype('float32')
            #print(x_train.shape,y_train.shape)
            yield (x_train, y_train)
        f.close()

img_rows, img_cols = 36, 64
input_shape = (img_rows, img_cols, 1)
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, 1)
#y_test = keras.utils.to_categorical(y_test, 1)
#time.sleep(10)
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

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=0.001))

model.fit_generator(generate_arrays_from_file("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/complete_for_training/augmented_data_FULL.csv",),
                    steps_per_epoch=1000, epochs=10)


#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          shuffle=True,
#          validation_data=(x_test, y_test))

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score)
#print(model.predict(x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)))
#print(y_train)


#model.save_weights("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/first_model_64x36.h5")