# import the necessary packages
import numpy as np
import cv2
import imutils
from imutils import paths
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

# convert the image to grayscale, blur it, and detect edges
video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/IMG_3161.MOV")
ok, frame = video.read()
frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 35, 125)

# find the contours in the edged image and keep the largest one;
# we'll assume that this is our piece of paper in the image
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

largest_area=0
for i in contours:
    area = cv2.contourArea(cnt)
    if (area>largest_area):
        largest_area=area
        largest_contour_index=i
        bounding_rect=cv2.boundingRect(contours[i])
rect=edged(bounding_rect).clone()
cv2.imshow('largest contour ',rect)
cv2.waitKey()
cv2.destroyAllWindows()
#cv2.imshow("huhu",cnts)
#cv2.waitKey(0)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
c = max(cnts, key=cv2.contourArea)
#cv2.imshow("huhu",cv2.minAreaRect(c))
print(cv2.minAreaRect())
# compute the bounding box of the of the paper region and return it
#cv2.imshow(cv2.minAreaRect(c))