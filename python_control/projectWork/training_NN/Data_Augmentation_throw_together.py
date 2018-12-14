import numpy as np
import cv2
import time
from os import listdir
from os.path import isfile, join

mypath="train_data/augmented_data"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
first=True
i=0
for file_name in onlyfiles:
    if first:
        aug_data = np.array(np.loadtxt(
            open(
                "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/" + file_name,
                "rb"),
            delimiter=",", skiprows=1))
        first=False
    else:

        aug_data=np.vstack((aug_data,np.array(np.loadtxt(
            open(
                "/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/" + file_name,
                "rb"),
            delimiter=",", skiprows=1))))
    print(i)
    i=i+1

np.savetxt("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/complete_for_training/augmented_data_FULL.csv", aug_data,
               delimiter=",", fmt='%.6f')






