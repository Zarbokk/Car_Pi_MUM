import numpy as np
import cv2
import time

def noisy(image):
  row,col= image.shape
  mean = 0
  var = 0.001
  sigma = var**0.5
  gauss = np.random.normal(mean,sigma,(row,col))
  gauss = gauss.reshape(row,col)
  noisy = image + gauss
  return noisy


#train_data=np.array(np.loadtxt(open("/Users/tim/OneDrive/Uni/StudienArbeit/Test_data/grey_image_data_sample.csv", "rb"), delimiter=",", skiprows=1))
train_data=np.array(np.loadtxt(open("F:/OneDrive/Uni/StudienArbeit/Car_Dataset/grey_image_data.csv", "rb"), delimiter=",", skiprows=1))
#print(train_data.shape)
#np.savetxt("E:/ML_MODELS/Data_for_AirSim/grey_image_data_augmented_sample.csv", train_data[0:2500,:], delimiter=",",fmt='%.6f')
#(144, 256)
#cv2.flip(gray_image,1)) makes the angle multiply by -1
#noisy(gray_image) adds gaussian noise
#cv2.warpAffine(gray_image,np.float32([[1,0,30],[0,1,0]]),(256,144)) 30=shift in x richtung 256 und 144 right size
#v2.resize(img,(64,36)) wäre gut für NN input
shift=np.asarray((10,20))
shift_angle=0.5
row,col=train_data.shape
print(row,col)
dimx=410
dimy=308

number_of_dp=4*shift.shape[0]+2
augmented_data=np.ones((row*number_of_dp,col))
print(augmented_data.shape)
for x in range(0,row):
    gray_image=train_data[x, 1:].reshape(dimy, dimx)/255
    cv2.imshow("Image Window", gray_image)
    cv2.waitKey(1)
    steering_angle=train_data[x, 0]
    #add noise
    gray_image_noisy=noisy(gray_image)
    for i in range(0,shift.shape[0]):
        x_shift=shift[i]
        steering_mod = steering_angle + x_shift / 100 * shift_angle
        gray_image_noisy_flatted=np.array(cv2.warpAffine(gray_image_noisy,np.float32([[1,0,x_shift],[0,1,0]]),(dimx,dimy))).flatten()
        augmented_data[x * number_of_dp + i ,:]=np.hstack([steering_mod,gray_image_noisy_flatted])

    for i in range(0,shift.shape[0]):
        x_shift=-shift[i]
        steering_mod = steering_angle + x_shift / 100 * shift_angle
        gray_image_noisy_flatted=np.array(cv2.warpAffine(gray_image_noisy,np.float32([[1,0,x_shift],[0,1,0]]),(dimx,dimy))).flatten()
        augmented_data[x*number_of_dp+i+shift.shape[0], ]=np.hstack([steering_mod,gray_image_noisy_flatted])
    gray_image_noisy_old=gray_image_noisy
    gray_image_noisy=cv2.flip(gray_image_noisy,1)
    steering_angle=steering_angle*-1
    for i in range(0, shift.shape[0]):
        x_shift = shift[i]
        steering_mod = steering_angle + x_shift / 100 * shift_angle
        gray_image_noisy_flatted = np.array(cv2.warpAffine(gray_image_noisy,np.float32([[1,0,x_shift],[0,1,0]]),(dimx,dimy))).flatten()
        augmented_data[x * number_of_dp + i + shift.shape[0]*2,] = np.hstack([steering_mod, gray_image_noisy_flatted])

    for i in range(0, shift.shape[0]):
        x_shift = -shift[i]
        steering_mod = steering_angle + x_shift / 100 * shift_angle
        gray_image_noisy_flatted = np.array(cv2.warpAffine(gray_image_noisy,np.float32([[1,0,x_shift],[0,1,0]]),(dimx,dimy))).flatten()
        augmented_data[x * number_of_dp + i + shift.shape[0]*3,] = np.hstack([steering_mod, gray_image_noisy_flatted])

    augmented_data[x * number_of_dp + shift.shape[0]*4,] = np.hstack([steering_angle*-1, np.array(gray_image_noisy_old).flatten()])
    augmented_data[x * number_of_dp + shift.shape[0]*4+1,] = np.hstack([steering_angle, np.array(gray_image_noisy).flatten()])
print("end")

# row,col=augmented_data.shape
# for x in range(0,row):
#     gray_image = augmented_data[x,1:].reshape(144, 256)
#     print(augmented_data[x,0])
#     cv2.imshow('image', gray_image)
#     cv2.waitKey(0)

#np.savetxt("E:/ML_MODELS/Data_for_AirSim/grey_image_data_augmented.csv", augmented_data, delimiter=",",fmt='%.6f')