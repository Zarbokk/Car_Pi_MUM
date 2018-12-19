import numpy as np
import cv2
import time
from os import listdir
from os.path import isfile, join


def noisy(image):
    row = image.shape
    mean = 0
    var = 0.001
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row))
    # gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def resize_image(frame):
    tmp = frame[1:].reshape(308, 410)
    rescaled = cv2.resize(tmp, (64, 36))
    # print(rescaled.shape)
    # cv2.imshow("Image Window", rescaled)
    # cv2.waitKey()
    return np.hstack([frame[0], rescaled.flatten()])


def brightness(image, brightness_in):
    dimx = 410
    dimy = 308
    image_solo = np.where((255 - image[1:] * 255) < brightness_in, 255, image[1:] * 255 + brightness_in) / 255
    return_value = np.hstack([image[0], noisy(image_solo)])
    return return_value


# train_data=np.array(np.loadtxt(open("/Users/tim/OneDrive/Uni/StudienArbeit/Test_data/gray_image_data_sample.csv", "rb"), delimiter=",", skiprows=1))
# train_data=np.array(np.loadtxt(open("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/gray_image_data.csv", "rb"), delimiter=",", skiprows=1))
# print(train_data.shape)
# np.savetxt("E:/ML_MODELS/Data_for_AirSim/gray_image_data_augmented_sample.csv", train_data[0:2500,:], delimiter=",",fmt='%.6f')
# (144, 256)
# cv2.flip(gray_image,1)) makes the angle multiply by -1
# noisy(gray_image) adds gaussian noise
# cv2.warpAffine(gray_image,np.float32([[1,0,30],[0,1,0]]),(256,144)) 30=shift in x richtung 256 und 144 right size
# v2.resize(img,(64,36)) waere gut fuer NN input
mypath = "train_data"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# print(len(onlyfiles))
dimx = 410
dimy = 308
j = 0
for file_name in onlyfiles:
    train_data = np.array(np.loadtxt(
        open("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/" + file_name, "rb"),
        delimiter=","))
    shift = np.asarray((range(0, 3))) * 8

    # print(shift.shape[0])
    shift_angle = 0.5
    row, col = train_data.shape
    # print(row,col)

    number_of_dp = 4 * shift.shape[0] + 2
    augmented_data = np.ones((row * number_of_dp, col))
    # print(augmented_data.shape)
    # print(row)
    for x in range(0, row):
        # print(max(shift))
        gray_image = train_data[x, 1:].reshape(dimy, dimx) / 255
        # cv2.imshow("Image Window", gray_image)
        # cv2.waitKey()

        # res = cv2.resize(grey_new, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        steering_angle = train_data[x, 0] / 29
        # add noise
        # gray_image_noisy = noisy(gray_image)
        # right moving
        for i in range(0, shift.shape[0]):
            x_shift = shift[i]
            steering_mod = steering_angle + x_shift / float(max(shift)) * shift_angle
            gray_image_noisy_flatted = np.array(
                cv2.warpAffine(gray_image, np.float32([[1, 0, x_shift], [0, 1, 0]]), (dimx, dimy))).flatten()
            augmented_data[x * number_of_dp + i, :] = np.hstack([steering_mod, gray_image_noisy_flatted])
        # left moving
        for i in range(0, shift.shape[0]):
            x_shift = -shift[i]
            steering_mod = steering_angle + x_shift / float(max(shift)) * shift_angle
            gray_image_noisy_flatted = np.array(
                cv2.warpAffine(gray_image, np.float32([[1, 0, x_shift], [0, 1, 0]]), (dimx, dimy))).flatten()
            augmented_data[x * number_of_dp + i + shift.shape[0],] = np.hstack([steering_mod, gray_image_noisy_flatted])

        gray_image_old = gray_image
        gray_image_flipped = cv2.flip(gray_image, 1)
        steering_angle = steering_angle * -1
        for i in range(0, shift.shape[0]):
            x_shift = shift[i]
            steering_mod = steering_angle + x_shift / float(max(shift)) * shift_angle
            gray_image_flatted = np.array(
                cv2.warpAffine(gray_image_flipped, np.float32([[1, 0, x_shift], [0, 1, 0]]), (dimx, dimy))).flatten()
            augmented_data[x * number_of_dp + i + shift.shape[0] * 2,] = np.hstack([steering_mod, gray_image_flatted])

        for i in range(0, shift.shape[0]):
            x_shift = -shift[i]
            steering_mod = steering_angle + x_shift / float(max(shift)) * shift_angle
            gray_image_noisy_flatted = np.array(
                cv2.warpAffine(gray_image_flipped, np.float32([[1, 0, x_shift], [0, 1, 0]]), (dimx, dimy))).flatten()
            augmented_data[x * number_of_dp + i + shift.shape[0] * 3,] = np.hstack(
                [steering_mod, gray_image_noisy_flatted])
        augmented_data[x * number_of_dp + shift.shape[0] * 4,] = np.hstack(
            [steering_angle * -1, np.array(gray_image_old).flatten()])
        augmented_data[x * number_of_dp + shift.shape[0] * 4 + 1,] = np.hstack(
            [steering_angle, np.array(gray_image_flipped).flatten()])
    # for i in range(0, augmented_data.shape[0]):
    #    cv2.imshow("Image Window", augmented_data[i, 1:].reshape(dimy, dimx))
    #    cv2.waitKey()
    # print("done")
    scaling = np.asarray((0, 5))
    # scaling = np.asarray((-15,-10,-5,0,5,10,15))
    # print(augmented_data.shape[0],scaling.shape[0], 64 * 36 + 1)
    # print(augmented_data.shape[0]*scaling.shape)
    contrast_data = np.ones((augmented_data.shape[0] * scaling.shape[0], 64 * 36 + 1))
    for i in range(0, augmented_data.shape[0]):
        for k in range(0, scaling.shape[0]):
            contrast_data[i * scaling.shape[0] + k,] = resize_image(brightness(augmented_data[i,], scaling[k]))
    # print(augmented_data.shape[0])

    # for i in range(0,contrast_data.shape[0]):
    #    cv2.imshow("Image Window", contrast_data[i,1:].reshape(36, 64))
    #    cv2.waitKey()

    # data_complete = np.ones((augmented_data.shape[0], 64 * 36 + 1))
    # print(data_complete.shape)
    # for i in range(0, augmented_data.shape[0]):
    #    data_complete[i,] = np.append(augmented_data[i, 0], resize_image(augmented_data[i, 1:]))
    np.savetxt(
        "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/augmented_" + file_name,
        contrast_data,
        delimiter=",", fmt='%.6f')
    print(j)
    j = j + 1
# row,col=augmented_data.shape
# for x in range(0,row):
#     gray_image = augmented_data[x,1:].reshape(144, 256)
#     print(augmented_data[x,0])
#     cv2.imshow('image', gray_image)
#     cv2.waitKey(0)

# np.savetxt("E:/ML_MODELS/Data_for_AirSim/gray_image_data_augmented.csv", augmented_data, delimiter=",",fmt='%.6f')
