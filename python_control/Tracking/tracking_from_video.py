import cv2
from car_scripts.red_dots_tracking_class import tracking_red_dots



largest_area=0
largest_contour_index=0
#video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/IMG_3161.MOV")
#video = cv2.VideoCapture("/home/tim/Downloads/IMG_3161.MOV")
video = cv2.VideoCapture("/home/tim/Dokumente/1280_32.avi")
#video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/1280_32.avi")
tracker = tracking_red_dots(960, 1280, 350, 900, 400, 960)
while(1):
    ok, frame = video.read()

    x_0, y_0, x_1, y_1 = tracker.get_red_pos(frame)
    circle = cv2.circle(frame, (x_1, y_1), 5, 120, -1)
    circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
    cv2.imshow("Image Window", circle)
    cv2.waitKey()



