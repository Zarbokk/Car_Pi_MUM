#! /usr/bin/env python
"""Class for Tracking via recognizing red shock absorbers."""
import cv2
import numpy as np


class tracking_red_dots:
    def __init__(self, height, width, x, w, y, h):
        # TODO: proper variable namings OR good explanation
        self.resize = 1
        self.x = x  # 350
        self.y = y  # 400
        self.w = w  # 900
        self.h = h  # 960
        self.first_done = False
        self.first_run_pos = [0, 0, 1, 1]
        self.height = height  # 960
        self.width = width  # 1280
        self.solidity_1 = 0.9
        self.solidity_0 = 0.9

        self.x_pos_old_0 = int((547 * width / 1280 - x) * self.resize)
        self.y_pos_old_0 = int((640 * height / 960 - y) * self.resize)
        self.x_pos_old_1 = int((700 * width / 1280 - x) * self.resize)
        self.y_pos_old_1 = int((640 * height / 960 - y) * self.resize)
        self.area_0 = 400*self.resize
        self.area_1 = 400*self.resize

    def get_red_pos(self, frame):
        """Get Red Position out of frame.

        (TODO) explain used routines:
        - erode:
        - dilate:

        """
        crop_img = frame[self.y:self.h, self.x:self.w]
        # crop_img = frame[320:321,:960]
        # cv2.imshow('crop_image', crop_img)
        # cv2.waitKey(1)
        frame = crop_img
        frame = cv2.resize(frame, (0, 0), fx=self.resize, fy=self.resize)

        # cv2.imshow('largest contour', frame)
        # cv2.waitKey()
        # r, g, b = cv2.split(frame)
        # cv2.imshow('largest contour', r)
        # cv2.waitKey()

        #print(self.x_pos_old_1, self.y_pos_old_1)
        #circle = cv2.circle(frame, (self.x_pos_old_1, self.y_pos_old_1), 5, 120, -1)
        #circle = cv2.circle(circle, (self.x_pos_old_0, self.y_pos_old_0), 5, 120, -1)
        #cv2.imshow('largest contour', circle)
        #cv2.waitKey()


        #frame = hsv = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
        frame = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        frame = hsv = cv2.GaussianBlur(frame, (11, 11), 0)
        #cv2.imshow('largest contour', hsv)
        #cv2.waitKey()

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([20, 255, 255])
        mask0 = cv2.inRange(frame, lower_red, upper_red)

        lower_red = np.array([160, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(frame, lower_red, upper_red)

        mask = mask0 + mask1
        #mask = cv2.erode(mask, None, iterations=2)
        #mask = cv2.dilate(mask, None, iterations=2)
        frame = mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        frame = cv2.dilate(frame, kernel)
        #cv2.imshow('largest contour',frame)
        #cv2.waitKey()
        image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        data_matrix = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            print(area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            distance_0 = np.sqrt(
                (self.x_pos_old_0 - x - w / 2)**2 +
                (self.y_pos_old_0 - y - h / 2)**2
            )
            distance_1 = np.sqrt(
                (self.x_pos_old_1 - x - w / 2)**2 +
                (self.y_pos_old_1 - y - h / 2)**2
            )
            newrow = [x, y, w, h, area, solidity, distance_0, distance_1]
            data_matrix = np.vstack([data_matrix, newrow])

        data_matrix = data_matrix[1:, :]
        # (TODO: improve comment) filter contours contained in specified area
        data_matrix = data_matrix[
            data_matrix[:, 4] > min(self.area_0, self.area_1) / 2, :]
        data_matrix = data_matrix[
            data_matrix[:, 4] < max(self.area_0, self.area_1) * 2, :]
        data_matrix = data_matrix[
            data_matrix[:, 5] > min(self.solidity_0, self.solidity_1) * 0.5, :]
        sorted_by_pos_0 = sorted(data_matrix[:, 6])
        sorted_by_pos_1 = sorted(data_matrix[:, 7])
        bottom_pos_0 = sorted_by_pos_0[0]
        bottom_pos_1 = sorted_by_pos_1[0]
        pos_0 = np.where(data_matrix[:, 6] == bottom_pos_0)
        pos_1 = np.where(data_matrix[:, 7] == bottom_pos_1)

        # print(pos_0,pos_1)
        # print(data_matrix.shape)

        if (pos_0[0] == pos_1[0]):
            ix = pos_0[0]  # use this now as index
            print("pos gleich")
            print(self.x_pos_old_0, self.y_pos_old_0)
            print(data_matrix[ix, 0] + data_matrix[ix, 2] / 2,
                  data_matrix[ix, 1] + data_matrix[ix, 3] / 2)
            print(self.x_pos_old_1, self.y_pos_old_1)
            print(data_matrix[ix, 6])
            if (data_matrix[ix, 6] < data_matrix[ix, 7]):
                # pos_0 naeher dran
                diff_x = (self.x_pos_old_0 -
                          data_matrix[ix, 0] - data_matrix[ix, 2] / 2)
                diff_y = (self.y_pos_old_0 -
                          data_matrix[ix, 1] - data_matrix[ix, 3] / 2)
                self.x_pos_old_1 -= diff_x
                self.y_pos_old_1 -= diff_y

                self.x_pos_old_0 = data_matrix[ix, 0] + data_matrix[ix, 2] / 2
                self.y_pos_old_0 = data_matrix[ix, 1] + data_matrix[ix, 3] / 2
                self.area_0 = data_matrix[ix, 4]
                self.solidity_0 = data_matrix[ix, 5]
            else:  # pos_1 naeher dran
                # TODO: das tritt nie ein oder?
                # oben checken wir auf
                ix = pos_1[0]
                diff_x = (self.x_pos_old_1 -
                          data_matrix[ix, 0] - data_matrix[ix, 2] / 2)
                diff_y = (self.y_pos_old_1 -
                          data_matrix[ix, 1] - data_matrix[ix, 3] / 2)
                self.x_pos_old_0 -= diff_x
                self.y_pos_old_0 -= diff_y

                self.x_pos_old_1 = data_matrix[ix, 0] + data_matrix[ix, 2] / 2
                self.y_pos_old_1 = data_matrix[ix, 1] + data_matrix[ix, 3] / 2
                self.area_1 = data_matrix[ix, 4]
                self.solidity_1 = data_matrix[ix, 5]
        else:
            ix1 = pos_1[0]
            ix0 = pos_0[0]
            if (
                    data_matrix[ix1, 7] < self.width * 0.05 and
                    data_matrix[ix0, 6] < self.width * 0.05
            ):
                a = [data_matrix[ix0, 0] - data_matrix[ix1, 0],
                     data_matrix[ix0, 1] - data_matrix[ix1, 1]]
                b = [-1, 0]
                angle = np.arccos(
                    (a[0] * b[0] + a[1] * b[1]) /
                    np.sqrt(a[0]**2 + a[1]**2) / np.sqrt(b[0]**2 + b[1]**2))
                # print(angle * 180 / 3.14159)
                # [x, y, w, h, area, solidity, distance_0, distance_1]
                if (angle * 180 / 3.14159 < 15):
                    self.x_pos_old_0 = data_matrix[ix0, 0] + data_matrix[ix0, 2] / 2
                    self.y_pos_old_0 = data_matrix[ix0, 1] + data_matrix[ix0, 3] / 2
                    self.area_0 = data_matrix[ix0, 4]
                    self.solidity_0 = data_matrix[ix0, 5]

                    self.x_pos_old_1 = data_matrix[ix1, 0] + data_matrix[ix1, 2] / 2
                    self.y_pos_old_1 = data_matrix[ix1, 1] + data_matrix[ix1, 3] / 2
                    self.area_1 = data_matrix[ix1, 4]
                    self.solidity_1 = data_matrix[ix1, 5]
                else:
                    if (data_matrix[ix0, 6] < data_matrix[ix1, 7]):
                        # pos_0 naeher drann
                        diff_x = self.x_pos_old_0 - data_matrix[ix0, 0] - data_matrix[ix0, 2] / 2
                        diff_y = self.y_pos_old_0 - data_matrix[ix0, 1] - data_matrix[ix0, 3] / 2
                        self.x_pos_old_1 -= diff_x
                        self.y_pos_old_1 -= diff_y

                        self.x_pos_old_0 = data_matrix[ix0, 0] + data_matrix[ix0, 2] / 2
                        self.y_pos_old_0 = data_matrix[ix0, 1] + data_matrix[ix0, 3] / 2
                        self.area_0 = data_matrix[ix0, 4]
                        self.solidity_0 = data_matrix[ix0, 5]
                    else:
                        # pos_1 naeher drann
                        diff_x = self.x_pos_old_1 - data_matrix[ix1, 0] - data_matrix[ix1, 2] / 2
                        diff_y = self.y_pos_old_1 - data_matrix[ix1, 1] - data_matrix[ix1, 3] / 2
                        self.x_pos_old_0 -= diff_x
                        self.y_pos_old_0 -= diff_y

                        self.x_pos_old_1 = data_matrix[ix1, 0] + data_matrix[ix1, 2] / 2
                        self.y_pos_old_1 = data_matrix[ix1, 1] + data_matrix[ix1, 3] / 2
                        self.area_1 = data_matrix[ix1, 4]
                        self.solidity_1 = data_matrix[ix1, 5]
        if (
            np.sqrt(
                (self.x_pos_old_1 - self.x_pos_old_0)**2 +
                (self.y_pos_old_1 - self.y_pos_old_0)**2) > self.width / 3
        ):
            self.solidity_1 = 0.9
            self.solidity_0 = 0.9
            self.x_pos_old_0 = self.first_run_pos[0]
            self.y_pos_old_0 = self.first_run_pos[1]
            self.x_pos_old_1 = self.first_run_pos[2]
            self.y_pos_old_1 = self.first_run_pos[3]
            self.area_0 = 400 * self.resize
            self.area_1 = 400 * self.resize


        circle = cv2.circle(frame, (self.x_pos_old_1, self.y_pos_old_1), 2, 120, -1)
        circle = cv2.circle(circle, (self.x_pos_old_0, self.y_pos_old_0), 2, 120, -1)
        cv2.imshow('whiteDots', circle)
        cv2.imshow('largest contour', hsv)
        cv2.imwrite("/home/tim/Dokumente/poster/crob_image_hsv.png", hsv)
        cv2.imwrite("/home/tim/Dokumente/poster/crob_image_dots.png", circle)
        cv2.waitKey(1)
        #cv2.waitKey()
        if not self.first_done:
            self.first_run_pos = [
                self.x_pos_old_0, self.y_pos_old_0,
                self.x_pos_old_1, self.y_pos_old_1]
            self.first_done = True
        return self.x_pos_old_0 / self.resize + self.x, self.y_pos_old_0 / self.resize + self.y, self.x_pos_old_1 / self.resize + self.x, self.y_pos_old_1 / self.resize + self.y
