import cv2
import numpy as np

video = cv2.VideoCapture("/home/tim/Dokumente/1280_32.avi")

while 1:
    ok, frame = video.read()
    #frame = cv2.GaussianBlur(frame, (11, 11), 0)
    frame = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv_coordinaten', hsv)
    cv2.waitKey(1)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(frame, lower_red, upper_red)
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(frame, lower_red, upper_red)
    mask = mask0 + mask1
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    frame = mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    frame = cv2.dilate(frame, kernel)

    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('contour', frame)
    cv2.waitKey()