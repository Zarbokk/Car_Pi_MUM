import cv2
import numpy as np

class tracking_red_dots:
    def __init__(self,height,width,x,w,y,h):
        self.resize = 1
        self.first_done = False
        self.first_run_pos=[0,0,1,1]
        self.height = height
        self.width = width
        self.solidity_1 = 0.9
        self.solidity_0 = 0.9
        self.x_pos_old_0 = 587*width/1280*self.resize
        self.y_pos_old_0 = 700*height/960*self.resize
        self.x_pos_old_1 = 700*width/1280*self.resize
        self.y_pos_old_1 = 700*height/960*self.resize
        self.area_0 = 400*self.resize
        self.area_1 = 400*self.resize
    def get_red_pos(self,frame):

        crop_img = frame[self.height/3:int(self.height/3*2.5), self.width/3:self.width/3*2]
        #cv2.imshow('crop_image', crop_img)
        #cv2.waitKey(1)
        frame=crop_img

        frame = cv2.resize(frame, (0, 0), fx=self.resize, fy=self.resize)



        #cv2.imshow('largest contour', frame)
        #cv2.waitKey()
        #r, g, b = cv2.split(frame)
        # cv2.imshow('largest contour', r)
        # cv2.waitKey()
        #circle = cv2.circle(frame, (self.x_pos_old_1, self.y_pos_old_1), 5, 120, -1)
        #circle = cv2.circle(circle, (self.x_pos_old_0, self.y_pos_old_0), 5, 120, -1)
        #cv2.imshow('largest contour', circle)
        #cv2.waitKey()
        frame = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow('largest contour', frame)
        #cv2.waitKey()
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
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
        #cv2.imshow('largest contour',frame)
        #cv2.waitKey()
        image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        simpleList = []
        data_matrix = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # print(area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            distance_0 = np.sqrt((self.x_pos_old_0 - x - w / 2) * (self.x_pos_old_0 - x - w / 2) + (self.y_pos_old_0 - y - h / 2) * (
                    self.y_pos_old_0 - y - h / 2))
            distance_1 = np.sqrt((self.x_pos_old_1 - x - w / 2) * (self.x_pos_old_1 - x - w / 2) + (self.y_pos_old_1 - y - h / 2) * (
                    self.y_pos_old_1 - y - h / 2))
            newrow = [x, y, w, h, area, solidity, distance_0, distance_1]
            data_matrix = np.vstack([data_matrix, newrow])
        data_matrix = data_matrix[1:, :]
        data_matrix = data_matrix[data_matrix[:, 4] > min(self.area_0, self.area_1) / 2, :]
        data_matrix = data_matrix[data_matrix[:, 4] < max(self.area_0, self.area_1) * 2, :]
        data_matrix = data_matrix[data_matrix[:, 5] > min(self.solidity_0, self.solidity_1) * 0.5, :]
        sorted_by_pos_0 = sorted(data_matrix[:, 6])
        sorted_by_pos_1 = sorted(data_matrix[:, 7])
        bottom_pos_0 = sorted_by_pos_0[0]
        bottom_pos_1 = sorted_by_pos_1[0]
        pos_0 = np.where(data_matrix[:, 6] == bottom_pos_0)
        pos_1 = np.where(data_matrix[:, 7] == bottom_pos_1)

        # print(pos_0,pos_1)
        # print(data_matrix.shape)

        if (pos_0[0] == pos_1[0]):
            print("pos gleich")
            print(self.x_pos_old_0, self.y_pos_old_0)
            print(data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2,
                  data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2)
            print(self.x_pos_old_1, self.y_pos_old_1)
            print(data_matrix[pos_1[0], 6])
            if (data_matrix[pos_0[0], 6] < data_matrix[pos_0[0], 7]):
                # pos_0 naeher drann
                diff_x = self.x_pos_old_0 - data_matrix[pos_0[0], 0] - data_matrix[pos_0[0], 2] / 2
                diff_y = self.y_pos_old_0 - data_matrix[pos_0[0], 1] - data_matrix[pos_0[0], 3] / 2
                self.x_pos_old_1 = self.x_pos_old_1 - diff_x
                self.y_pos_old_1 = self.y_pos_old_1 - diff_y

                self.x_pos_old_0 = data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2
                self.y_pos_old_0 = data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2
                self.area_0 = data_matrix[pos_0[0], 4]
                self.solidity_0 = data_matrix[pos_0[0], 5]
            else:
                # pos_1 naeher drann
                diff_x = self.x_pos_old_1 - data_matrix[pos_1[0], 0] - data_matrix[pos_1[0], 2] / 2
                diff_y = self.y_pos_old_1 - data_matrix[pos_1[0], 1] - data_matrix[pos_1[0], 3] / 2
                self.x_pos_old_0 = self.x_pos_old_0 - diff_x
                self.y_pos_old_0 = self.y_pos_old_0 - diff_y

                self.x_pos_old_1 = data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2] / 2
                self.y_pos_old_1 = data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3] / 2
                self.area_1 = data_matrix[pos_1[0], 4]
                self.solidity_1 = data_matrix[pos_1[0], 5]
        else:
            #print("hello")
            if (data_matrix[pos_1[0], 7] < self.width * 0.05 and data_matrix[pos_0[0], 6] < self.width * 0.05):
                a = [data_matrix[pos_0[0], 0] - data_matrix[pos_1[0], 0],
                     data_matrix[pos_0[0], 1] - data_matrix[pos_1[0], 1]]
                b = [-1, 0]
                angle = np.arccos(
                    (a[0] * b[0] + a[1] * b[1]) / np.sqrt(a[0] * a[0] + a[1] * a[1]) / np.sqrt(
                        b[0] * b[0] + b[1] * b[1]))
                #print(angle * 180 / 3.14159)
                #[x, y, w, h, area, solidity, distance_0, distance_1]
                if (angle * 180 / 3.14159 < 15):
                    #print(angle * 180 / 3.14159)
                    self.x_pos_old_0 = data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2
                    self.y_pos_old_0 = data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2
                    self.area_0 = data_matrix[pos_0[0], 4]
                    self.solidity_0 = data_matrix[pos_0[0], 5]

                    self.x_pos_old_1 = data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2] / 2
                    self.y_pos_old_1 = data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3] / 2
                    self.area_1 = data_matrix[pos_1[0], 4]
                    self.solidity_1 = data_matrix[pos_1[0], 5]
                else:
                    if (data_matrix[pos_0[0], 6] < data_matrix[pos_1[0], 7]):
                        # pos_0 naeher drann
                        diff_x = self.x_pos_old_0 - data_matrix[pos_0[0], 0] - data_matrix[pos_0[0], 2] / 2
                        diff_y = self.y_pos_old_0 - data_matrix[pos_0[0], 1] - data_matrix[pos_0[0], 3] / 2
                        self.x_pos_old_1 = self.x_pos_old_1 - diff_x
                        self.y_pos_old_1 = self.y_pos_old_1 - diff_y

                        self.x_pos_old_0 = data_matrix[pos_0[0], 0] + data_matrix[pos_0[0], 2] / 2
                        self.y_pos_old_0 = data_matrix[pos_0[0], 1] + data_matrix[pos_0[0], 3] / 2
                        self.area_0 = data_matrix[pos_0[0], 4]
                        self.solidity_0 = data_matrix[pos_0[0], 5]
                    else:
                        # pos_1 naeher drann
                        diff_x = self.x_pos_old_1 - data_matrix[pos_1[0], 0] - data_matrix[pos_1[0], 2] / 2
                        diff_y = self.y_pos_old_1 - data_matrix[pos_1[0], 1] - data_matrix[pos_1[0], 3] / 2
                        self.x_pos_old_0 = self.x_pos_old_0 - diff_x
                        self.y_pos_old_0 = self.y_pos_old_0 - diff_y

                        self.x_pos_old_1 = data_matrix[pos_1[0], 0] + data_matrix[pos_1[0], 2] / 2
                        self.y_pos_old_1 = data_matrix[pos_1[0], 1] + data_matrix[pos_1[0], 3] / 2
                        self.area_1 = data_matrix[pos_1[0], 4]
                        self.solidity_1 = data_matrix[pos_1[0], 5]
        if (np.sqrt((self.x_pos_old_1 - self.x_pos_old_0) * (self.x_pos_old_1 - self.x_pos_old_0) + (self.y_pos_old_1 - self.y_pos_old_0) * (
                self.y_pos_old_1 - self.y_pos_old_0)) > self.width / 3):
            self.solidity_1 = 0.9
            self.solidity_0 = 0.9
            self.x_pos_old_0 = self.first_run_pos[0]
            self.y_pos_old_0 = self.first_run_pos[1]
            self.x_pos_old_1 = self.first_run_pos[2]
            self.y_pos_old_1 = self.first_run_pos[3]
            self.area_0 = 400*self.resize
            self.area_1 = 400*self.resize

        circle = cv2.circle(frame, (self.x_pos_old_1, self.y_pos_old_1), 5, 120, -1)
        circle = cv2.circle(circle, (self.x_pos_old_0, self.y_pos_old_0), 5, 120, -1)
        cv2.imshow('largest contour', circle)
        #print(self.area_0)
        #cv2.waitKey()
        if not self.first_done:
            self.first_run_pos=[self.x_pos_old_0,self.y_pos_old_0,self.x_pos_old_1,self.y_pos_old_1]
            self.first_done = True
        return self.x_pos_old_0/self.resize,self.y_pos_old_0/self.resize,self.x_pos_old_1/self.resize,self.y_pos_old_1/self.resize





k=3
print(k)