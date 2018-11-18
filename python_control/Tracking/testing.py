import cv2
import numpy as np
largest_area=0;
largest_contour_index=0
video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/IMG_3161.MOV")
ok, frame = video.read()
frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
img =frame
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray = cv2.GaussianBlur(imgray, (5, 5),1)
edged = cv2.Canny(imgray, 35, 100)

image, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if (area>largest_area):
        largest_area=area
        largest_contour_index=cnt
        x, y, w, h = cv2.boundingRect(cnt)
        #bounding_rect=cv2.boundingRect(cnt)
#print([bounding_rect[2],bounding_rect[3]])
print(largest_area)

#edged = cv2.rectangle(image, (x, y), (x + w, y + h), 100, 2)
print(edged.shape)

cv2.drawContours(edged, contours, -1, 255, 3)
cv2.imshow('largest contour',edged)
cv2.waitKey()

#print(largest_contour_index)
#rect=img([[bounding_rect[2],bounding_rect[3]]])
#cv2.imshow('largest contour ',rect)
#cv2.waitKey()
#cv2.destroyAllWindows()