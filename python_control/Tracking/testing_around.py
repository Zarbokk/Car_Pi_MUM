import cv2
import numpy as np




largest_area=0
largest_contour_index=0
#video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/IMG_3161.MOV")
#video = cv2.VideoCapture("/home/tim/Downloads/IMG_3161.MOV")
#video = cv2.VideoCapture("/home/tim/Dokumente/Video_car_find.avi")
video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/1280_32.avi")

while(1):
    ok, frame = video.read()

    frame = cv2.applyColorMap(frame, cv2.COLORMAP_RAINBOW)

    cv2.imshow('largest contour', frame)
    cv2.waitKey()

#frame=cv2.cvCvtColor(imageBgr, imageHsv, CV_RGB2HSV);
#frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
#img =frame
#cv2.imshow('largest contour', cl1)
cv2.imshow('R-RGB',im_color)
cv2.waitKey()
frame = cv2.GaussianBlur(frame, (11, 11), 0)
frame = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow('largest contour', hsv)
cv2.waitKey()
# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
# upper mask (170-180)
lower_red = np.array([0,0,0])
upper_red = np.array([180,150,150])
mask1 = cv2.inRange(frame, lower_red, upper_red)
mask = mask1

mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
frame = mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
frame = cv2.dilate(frame, kernel)
cv2.imshow('largest contour',frame)
cv2.waitKey()