import cv2
import numpy as np
class contour_data():
    def __init__(self, x, y, w,h,area,solidity):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
        self.solidity = solidity





largest_area=0;
largest_contour_index=0
video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/IMG_3161.MOV")
ok, frame = video.read()


while(1):
    ok, frame = video.read()
    #frame=cv2.cvCvtColor(imageBgr, imageHsv, CV_RGB2HSV);

    #frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    #img =frame
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    frame = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(frame, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(frame, lower_red, upper_red)
    mask = mask0 + mask1

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    frame = mask
    #frame = cv2.GaussianBlur(frame, (5, 5),1)
    #frame = cv2.Canny(frame, 35, 100)

    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    simpleList = []
    data_matrix=np.matrix('0 1 2 3 4 5')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        newrow = [x, y, w,h,area,solidity]
        data_matrix = np.vstack([data_matrix, newrow])
        #break
    #print(data_matrix.shape)
    data_matrix=data_matrix[1:,:]
    #print(data_matrix[:,1].shape)
    sorted_by_pos=sorted(data_matrix[:,1])
    bottom_pos=sorted_by_pos[-1]
    bottom_pos2 = sorted_by_pos[-2]
    pos=np.where(data_matrix == bottom_pos2)
    #print(pos)
    #print(data_matrix[pos[0],pos[1]])
    print(data_matrix[pos[0],:])
    edged = cv2.rectangle(frame, (data_matrix[pos[0],0], data_matrix[pos[0],1]), (data_matrix[pos[0],0] + data_matrix[pos[0],2], data_matrix[pos[0],1] + data_matrix[pos[0],3]), 100, 2)
    #for x in simpleList:

    #edged = cv2.rectangle(image, (x, y), (x + w, y + h), 100, 2)
    #print(edged.shape)
    #cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    print ("huhu")
    #cv2.drawContours(edged, contours, -1, 255, 3)
    cv2.imshow('largest contour',edged)
    cv2.waitKey()

#print(largest_contour_index)
#rect=img([[bounding_rect[2],bounding_rect[3]]])
#cv2.imshow('largest contour ',rect)
#cv2.waitKey()
#cv2.destroyAllWindows()