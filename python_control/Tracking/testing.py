import cv2
import numpy as np




largest_area=0
largest_contour_index=0
video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/IMG_3161.MOV")
ok, frame = video.read()
solidity_1 = 0.9
solidity_0 = 0.9
x_pos_old_0 = 500
y_pos_old_0 = 1000
x_pos_old_1 = 1000
y_pos_old_1 = 1000
area_0 = 700
area_1 = 700
while(1):
    ok, frame = video.read()
    #frame=cv2.cvCvtColor(imageBgr, imageHsv, CV_RGB2HSV);

    frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    frame = cv2.dilate(frame, kernel)
    cv2.imshow('largest contour',frame)
    cv2.waitKey()
    #frame = cv2.GaussianBlur(frame, (5, 5),1)
    #frame = cv2.Canny(frame, 35, 100)

    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    simpleList = []
    data_matrix=np.array([0 ,1, 2, 3, 4, 5, 6, 7])
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        print(area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        distance_0 = np.sqrt((x_pos_old_0-x-w/2)*(x_pos_old_0-x-w/2)+(y_pos_old_0-y-h/2)*(y_pos_old_0-y-h/2))
        distance_1 = np.sqrt((x_pos_old_1-x-w/2)*(x_pos_old_1-x-w/2)+(y_pos_old_1-y-h/2)*(y_pos_old_1-y-h/2))
        newrow = [x, y, w,h,area,solidity,distance_0,distance_1]
        data_matrix = np.vstack([data_matrix, newrow])
    data_matrix = data_matrix[1:, :]
    data_matrix=data_matrix[ data_matrix[:,4] > min(area_0,area_1)/2,:]
    data_matrix = data_matrix[data_matrix[:, 4] < max(area_0,area_1)*2, :]
    data_matrix = data_matrix[data_matrix[:, 5] > min(solidity_0,solidity_1)*0.5, :]
    print(data_matrix.shape)
    sorted_by_pos_0 = sorted(data_matrix[:, 6])
    sorted_by_pos_1 = sorted(data_matrix[:, 7])
    bottom_pos_0 = sorted_by_pos_0[0]
    bottom_pos_1 = sorted_by_pos_1[0]
    pos_0 = np.where(data_matrix[:,6] == bottom_pos_0)
    pos_1 = np.where(data_matrix[:,7] == bottom_pos_1)
    print(data_matrix.shape)
    print(np.asarray(pos_1).size)
    print(data_matrix[pos_0[0], :])
    #print(data_matrix[pos[0],pos[1]])
    #print(data_matrix[pos[0],:])
    edged = cv2.rectangle(frame, (data_matrix[pos_0[0], 0], data_matrix[pos_0[0], 1]), (
    data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2],
    data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3]), 100, 2)

    edged = cv2.rectangle(edged, (data_matrix[pos_1[0], 0], data_matrix[pos_1[0], 1]), (
    data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2],
    data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3]), 100, 2)

    x_pos_old_0=data_matrix[pos_0[0], 0]+data_matrix[pos_0[0], 2]/2
    y_pos_old_0 = data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2
    area_0 = data_matrix[pos_0[0], 4]
    solidity_0 = data_matrix[pos_0[0], 5]

    x_pos_old_1=data_matrix[pos_1[0], 0]+data_matrix[pos_1[0], 2]/2
    y_pos_old_1 = data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3] / 2
    area_1 = data_matrix[pos_1[0], 4]
    solidity_1 = data_matrix[pos_1[0], 5]
    #edged = cv2.rectangle(frame, (data_matrix[pos_0[1][0],0], data_matrix[pos_0[1][0],1]), (data_matrix[pos_0[1][0],0] + data_matrix[pos_0[1][0],2], data_matrix[pos_0[1][0],1] + data_matrix[pos_0[1][0],3]), 100, 2)
    #edged = cv2.rectangle(edged, (data_matrix[pos_1[0], 0], data_matrix[pos_1[0], 1]), (data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2], data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3]), 100, 2)
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