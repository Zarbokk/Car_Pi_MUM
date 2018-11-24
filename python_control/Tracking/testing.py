import cv2
import numpy as np




largest_area=0
largest_contour_index=0
#video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/IMG_3161.MOV")
#video = cv2.VideoCapture("/home/tim/Downloads/IMG_3161.MOV")
#video = cv2.VideoCapture("/home/tim/Dokumente/Video_car_find.avi")
video = cv2.VideoCapture("F:/OneDrive/Uni/StudienArbeit/Auto_Gruppe/Tracking_Auto/1280_32.avi")
#ok, frame = video.read()


#cv2.imshow("Original image", frame)
#cv2.waitKey()
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#b, g, r = cv2.split(frame)
#red = cv2.multiply(cv2.equalizeHist(r),1)
#green = cv2.multiply(cv2.equalizeHist(g),0.5)
#blue = cv2.multiply(cv2.equalizeHist(b),0.5)
#img_output = cv2.merge((blue, green, red))
#cv2.imwrite('Increased_Contrast.jpg', img3)
#cv2.imshow('Increased contrast', img_output)
#cv2.waitKey()

ok, frame = video.read()
height = np.size(frame, 0)
width = np.size(frame, 1)
solidity_1 = 0.9
solidity_0 = 0.9
x_pos_old_0 = 647
y_pos_old_0 = 550
x_pos_old_1 = 750
y_pos_old_1 = 550
area_0 = 400
area_1 = 400


while(1):

    frame = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow('largest contour', frame)
    #cv2.waitKey()
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    lower_red = np.array([0,50,50])
    upper_red = np.array([25,255,255])
    mask0 = cv2.inRange(frame, lower_red, upper_red)
    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(frame, lower_red, upper_red)
    mask = mask0 + mask1
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    frame = mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    frame = cv2.dilate(frame, kernel)
    #cv2.imshow('largest contour',frame)
    #cv2.waitKey()
    image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    simpleList = []
    data_matrix=np.array([0 ,1, 2, 3, 4, 5, 6, 7])
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        #print(area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        distance_0 = np.sqrt((x_pos_old_0-x-w/2)*(x_pos_old_0-x-w/2)+(y_pos_old_0-y-h/2)*(y_pos_old_0-y-h/2))
        distance_1 = np.sqrt((x_pos_old_1-x-w/2)*(x_pos_old_1-x-w/2)+(y_pos_old_1-y-h/2)*(y_pos_old_1-y-h/2))
        newrow = [x, y, w,h,area,solidity,distance_0,distance_1]
        data_matrix = np.vstack([data_matrix, newrow])
    data_matrix = data_matrix[1:, :]
    data_matrix = data_matrix[data_matrix[:, 4] > min(area_0,area_1)/2, :]
    data_matrix = data_matrix[data_matrix[:, 4] < max(area_0,area_1)*2, :]
    data_matrix = data_matrix[data_matrix[:, 5] > min(solidity_0,solidity_1)*0.5, :]
    sorted_by_pos_0 = sorted(data_matrix[:, 6])
    sorted_by_pos_1 = sorted(data_matrix[:, 7])
    bottom_pos_0 = sorted_by_pos_0[0]
    bottom_pos_1 = sorted_by_pos_1[0]
    pos_0 = np.where(data_matrix[:,6] == bottom_pos_0)
    pos_1 = np.where(data_matrix[:,7] == bottom_pos_1)

    #print(pos_0,pos_1)
    #print(data_matrix.shape)

    if(pos_0[0] == pos_1[0]):
        print("pos gleich")
        print(x_pos_old_0,y_pos_old_0)
        print(data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2,data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2)
        print(x_pos_old_1, y_pos_old_1)
        print(data_matrix[pos_1[0], 6])
        if(data_matrix[pos_0[0], 6]<data_matrix[pos_0[0], 7]):
            #pos_0 naeher drann
            diff_x = x_pos_old_0 - data_matrix[pos_0[0], 0] - data_matrix[pos_0[0], 2] / 2
            diff_y = y_pos_old_0 - data_matrix[pos_0[0], 1] - data_matrix[pos_0[0], 3] / 2
            x_pos_old_1 = x_pos_old_1 - diff_x
            y_pos_old_1 = y_pos_old_1 - diff_y



            x_pos_old_0 = data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2
            y_pos_old_0 = data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2
            area_0 = data_matrix[pos_0[0], 4]
            solidity_0 = data_matrix[pos_0[0], 5]
        else:
            #pos_1 naeher drann
            diff_x = x_pos_old_1 - data_matrix[pos_1[0], 0] - data_matrix[pos_1[0], 2] / 2
            diff_y = y_pos_old_1 - data_matrix[pos_1[0], 1] - data_matrix[pos_1[0], 3] / 2
            x_pos_old_0 = x_pos_old_0 - diff_x
            y_pos_old_0 = y_pos_old_0 - diff_y


            x_pos_old_1 = data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2] / 2
            y_pos_old_1 = data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3] / 2
            area_1 = data_matrix[pos_1[0], 4]
            solidity_1 = data_matrix[pos_1[0], 5]
    else:
        print("hello")
        if (data_matrix[pos_1[0], 7] < width * 0.1 and data_matrix[pos_0[0], 6] < width * 0.1):
            a = [data_matrix[pos_0[0], 0] - data_matrix[pos_1[0], 0],
                 data_matrix[pos_0[0], 1] - data_matrix[pos_1[0], 1]]
            b = [-1, 0]
            angle = np.arccos(
                (a[0] * b[0] + a[1] * b[1]) / np.sqrt(a[0] * a[0] + a[1] * a[1]) / np.sqrt(b[0] * b[0] + b[1] * b[1]))
            print(angle * 180 / 3.14159)
            if (angle * 180 / 3.14159 < 20):
                print(angle * 180 / 3.14159)
                x_pos_old_0 = data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2
                y_pos_old_0 = data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2
                area_0 = data_matrix[pos_0[0], 4]
                solidity_0 = data_matrix[pos_0[0], 5]

                x_pos_old_1 = data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2] / 2
                y_pos_old_1 = data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3] / 2
                area_1 = data_matrix[pos_1[0], 4]
                solidity_1 = data_matrix[pos_1[0], 5]
            else:
                if (data_matrix[pos_0[0], 6] < data_matrix[pos_0[0], 7]):
                    # pos_0 naeher drann
                    diff_x = x_pos_old_0 - data_matrix[pos_0[0], 0] - data_matrix[pos_0[0], 2] / 2
                    diff_y = y_pos_old_0 - data_matrix[pos_0[0], 1] - data_matrix[pos_0[0], 3] / 2
                    x_pos_old_1 = x_pos_old_1 - diff_x
                    y_pos_old_1 = y_pos_old_1 - diff_y

                    x_pos_old_0 = data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2
                    y_pos_old_0 = data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2
                    area_0 = data_matrix[pos_0[0], 4]
                    solidity_0 = data_matrix[pos_0[0], 5]
                else:
                    # pos_1 naeher drann
                    diff_x = x_pos_old_1 - data_matrix[pos_1[0], 0] - data_matrix[pos_1[0], 2] / 2
                    diff_y = y_pos_old_1 - data_matrix[pos_1[0], 1] - data_matrix[pos_1[0], 3] / 2
                    x_pos_old_0 = x_pos_old_0 - diff_x
                    y_pos_old_0 = y_pos_old_0 - diff_y

                    x_pos_old_1 = data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2] / 2
                    y_pos_old_1 = data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3] / 2
                    area_1 = data_matrix[pos_1[0], 4]
                    solidity_1 = data_matrix[pos_1[0], 5]
    if(np.sqrt((x_pos_old_1-x_pos_old_0)*(x_pos_old_1-x_pos_old_0)+(y_pos_old_1-y_pos_old_0)*(y_pos_old_1-y_pos_old_0))>width/2):
        solidity_1 = 0.9
        solidity_0 = 0.9
        x_pos_old_0 = 647
        y_pos_old_0 = 500
        x_pos_old_1 = 960
        y_pos_old_1 = 500
        area_0 = 400
        area_1 = 400










    #print(x_pos_old_0,y_pos_old_0)
    #print(x_pos_old_1, y_pos_old_1)


    #edged = cv2.rectangle(frame, (data_matrix[pos_0[0], 0], data_matrix[pos_0[0], 1]), (
    #    data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2],
    #    data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3]), 100, 2)
    #edged = cv2.rectangle(edged, (data_matrix[pos_1[0], 0], data_matrix[pos_1[0], 1]), (
    #    data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2],
    #    data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3]), 100, 2)
    circle= cv2.circle(frame, (x_pos_old_1, y_pos_old_1), 5, 120, -1)
    circle = cv2.circle(circle, (x_pos_old_0, y_pos_old_0), 5, 120, -1)
    cv2.imshow('largest contour',circle)
    cv2.waitKey()
    ok, frame = video.read()




k=3
print(k)