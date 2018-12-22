import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import cv2
batch_size = 32
epochs = 20


train_data = np.array(np.loadtxt(
            open(
                "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/complete_for_training/augmented_data_FULL.csv",
                "rb"),
            delimiter=","))
#train_data = np.array(np.loadtxt(
#            open(
#                "F:/OneDrive/Uni/StudienArbeit/Car_Dataset/augmented_data_FULL.csv",
#                "rb"),
#            delimiter=","))
print("hallo")
time.sleep(10)
#train_data=np.array(np.loadtxt(open("E:/ML_MODELS/Data_for_AirSim/grey_image_data_augmented.csv", "rb"), delimiter=",", skiprows=1))
#train_data=np.array(np.loadtxt(open("/Users/tim/OneDrive/Uni/StudienArbeit/Test_data/grey_image_data_augmented_sample.csv", "rb"), delimiter=",", skiprows=1))

nrows=train_data.shape[0]
# while True:
#     gray_image = train_data[k, 1:].reshape((144, 256))
#     k=k+1
#     print(k)
#     cv2.imshow('image', gray_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
nrows_train=int(nrows*0.9)
#nrows_train=10
print(nrows_train)
print(train_data.shape)
x_train=train_data[0:nrows_train,1:]
y_train=train_data[0:nrows_train,0]
x_test=train_data[nrows_train+1:,1:]
y_test=train_data[nrows_train+1:,0]
# input image dimensions
img_rows, img_cols = 36, 64
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_test[0,:,:,:])
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print('x_train shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(y_train.shape)
print(y_test.shape)
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, 1)
#y_test = keras.utils.to_categorical(y_test, 1)
time.sleep(10)
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(50, activation='relu'))
#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=0.001))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score)
#print('Test accuracy:', score[1])
print(model.predict(x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)))
print(y_train)
#print(model.predict(x_train[2,].reshape(1, img_rows, img_cols, 1)))
#print(y_train[2])
#print(model.predict(x_train[5,].reshape(1, img_rows, img_cols, 1)))
#print(y_train[5])
#print(model.predict(x_train[10,].reshape(1, img_rows, img_cols, 1)))
#print(y_train[10])

model.save_weights("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/models/third_model_64x36.h5")