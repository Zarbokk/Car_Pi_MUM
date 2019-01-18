#! /usr/bin/env python
"""Class for Tracking via recognizing red shock absorbers."""
import cv2
import numpy as np


class tracking_red_dots:
    def __init__(self, height, width, scaling=1, area_dots=300, solidarity=0.9):
        # TODO: proper variable namings OR good explanation
        self.resize_image_perf = scaling
        self.first_done = False
        self.first_run_pos = [0, 0, 1, 1]
        self.height = height  # 960
        self.width = width  # 1280
        self.solidity_1 = solidarity
        self.solidity_0 = solidarity
        self.area_0 = area_dots * self.resize_image_perf
        self.area_1 = area_dots * self.resize_image_perf

    def get_red_pos(self, frame, x_0, y_0, x_1, y_1, percentage_perf=0.15):
        x_0 = int(x_0)
        x_1 = int(x_1)
        y_0 = int(y_0)
        y_1 = int(y_1)
        #
        # circle = cv2.circle(frame.copy(), (x_1, y_1), 5, 120, -1)
        # circle = cv2.circle(circle, (x_0, y_0), 5, 120, -1)
        # cv2.imshow('testshit', circle)
        # cv2.waitKey(1)
        """Get Red Position out of frame.

        (TODO) explain used routines:
        - erode:
        - dilate:

        """
        # print(min(x_0, x_1) - self.height * percentage_perf)
        y_crop = int(min(y_0, y_1) - self.height * percentage_perf)
        h_crop = int(max(y_0, y_1) + self.height * percentage_perf)
        x_crop = int(min(x_0, x_1) - self.width * percentage_perf)
        w_crop = int(max(x_0, x_1) + self.width * percentage_perf)

        if y_crop < 0:
            y_crop = 0
        if x_crop < 0:
            x_crop = 0
        if h_crop > self.height:
            h_crop = self.height
        if w_crop > self.width:
            w_crop = self.width
        # print(y_crop, h_crop, x_crop, w_crop)
        crop_img = frame[y_crop:h_crop, x_crop:w_crop]
        save_img_between = crop_img.copy()
        x_0 = x_0 - x_crop
        y_0 = y_0 - y_crop
        x_1 = x_1 - x_crop
        y_1 = y_1 - y_crop
        # crop_img = frame[320:321,:960]
        # cv2.imshow('crop_image', crop_img)
        # cv2.waitKey(1)
        frame = crop_img
        frame = cv2.resize(frame, (0, 0), fx=self.resize_image_perf, fy=self.resize_image_perf)
        # frame = hsv = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv', frame)
        # cv2.waitKey(1)
        # frame = cv2.GaussianBlur(frame, (11, 11), 2)
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        # cv2.imshow('hsv', frame)
        # cv2.waitKey()

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([20, 255, 255])
        mask0 = cv2.inRange(frame, lower_red, upper_red)

        lower_red = np.array([160, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(frame, lower_red, upper_red)

        mask = mask0 + mask1
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        frame = mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        frame = cv2.dilate(frame, kernel)
        # cv2.imshow('largest contour', frame)
        # cv2.waitKey()
        image, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        data_matrix = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # print(area)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            distance_0 = np.sqrt(
                (x_0 - x - w / 2) ** 2 +
                (y_0 - y - h / 2) ** 2
            )
            distance_1 = np.sqrt(
                (x_1 - x - w / 2) ** 2 +
                (y_1 - y - h / 2) ** 2
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
            # print("pos gleich")
            # print(x_0, y_0)
            # print(data_matrix[ix, 0] + data_matrix[ix, 2] / 2,
            #       data_matrix[ix, 1] + data_matrix[ix, 3] / 2)
            # print(x_1, y_1)
            # print(data_matrix[ix, 6])
            if (data_matrix[ix, 6] < data_matrix[ix, 7]):
                # pos_0 naeher dran
                diff_x = (x_0 -
                          data_matrix[ix, 0] - data_matrix[ix, 2] / 2)
                diff_y = (y_0 -
                          data_matrix[ix, 1] - data_matrix[ix, 3] / 2)
                x_1 -= diff_x
                y_1 -= diff_y

                x_0 = data_matrix[ix, 0] + data_matrix[ix, 2] / 2
                y_0 = data_matrix[ix, 1] + data_matrix[ix, 3] / 2
                self.area_0 = data_matrix[ix, 4]
                self.solidity_0 = data_matrix[ix, 5]
            else:  # pos_1 naeher dran
                # TODO: das tritt nie ein oder?
                # oben checken wir auf
                ix = pos_1[0]
                diff_x = (x_1 -
                          data_matrix[ix, 0] - data_matrix[ix, 2] / 2)
                diff_y = (y_1 -
                          data_matrix[ix, 1] - data_matrix[ix, 3] / 2)
                x_0 -= diff_x
                y_0 -= diff_y

                x_1 = data_matrix[ix, 0] + data_matrix[ix, 2] / 2
                y_1 = data_matrix[ix, 1] + data_matrix[ix, 3] / 2
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
                    np.sqrt(a[0] ** 2 + a[1] ** 2) / np.sqrt(b[0] ** 2 + b[1] ** 2))
                # print(angle * 180 / 3.14159)
                # [x, y, w, h, area, solidity, distance_0, distance_1]
                if (angle * 180 / 3.14159 < 15):
                    x_0 = data_matrix[ix0, 0] + data_matrix[ix0, 2] / 2
                    y_0 = data_matrix[ix0, 1] + data_matrix[ix0, 3] / 2
                    self.area_0 = data_matrix[ix0, 4]
                    self.solidity_0 = data_matrix[ix0, 5]

                    x_1 = data_matrix[ix1, 0] + data_matrix[ix1, 2] / 2
                    y_1 = data_matrix[ix1, 1] + data_matrix[ix1, 3] / 2
                    self.area_1 = data_matrix[ix1, 4]
                    self.solidity_1 = data_matrix[ix1, 5]
                else:
                    if (data_matrix[ix0, 6] < data_matrix[ix1, 7]):
                        # pos_0 naeher drann
                        diff_x = x_0 - data_matrix[ix0, 0] - data_matrix[ix0, 2] / 2
                        diff_y = y_0 - data_matrix[ix0, 1] - data_matrix[ix0, 3] / 2
                        x_1 -= diff_x
                        y_1 -= diff_y

                        x_0 = data_matrix[ix0, 0] + data_matrix[ix0, 2] / 2
                        y_0 = data_matrix[ix0, 1] + data_matrix[ix0, 3] / 2
                        self.area_0 = data_matrix[ix0, 4]
                        self.solidity_0 = data_matrix[ix0, 5]
                    else:
                        # pos_1 naeher drann
                        diff_x = x_1 - data_matrix[ix1, 0] - data_matrix[ix1, 2] / 2
                        diff_y = y_1 - data_matrix[ix1, 1] - data_matrix[ix1, 3] / 2
                        x_0 -= diff_x
                        y_0 -= diff_y

                        x_1 = data_matrix[ix1, 0] + data_matrix[ix1, 2] / 2
                        y_1 = data_matrix[ix1, 1] + data_matrix[ix1, 3] / 2
                        self.area_1 = data_matrix[ix1, 4]
                        self.solidity_1 = data_matrix[ix1, 5]
        if (
                np.sqrt(
                    (x_1 - x_0) ** 2 +
                    (y_1 - y_0) ** 2) > self.width / 3
        ):
            self.solidity_1 = 0.9
            self.solidity_0 = 0.9
            x_0 = self.first_run_pos[0]
            y_0 = self.first_run_pos[1]
            x_1 = self.first_run_pos[2]
            y_1 = self.first_run_pos[3]
            self.area_0 = 400 * self.resize_image_perf
            self.area_1 = 400 * self.resize_image_perf

        # print(x_0, x_crop)

        # circle = cv2.circle(frame.copy(), (x_1, y_1), 2, 120, -1)
        # circle = cv2.circle(circle, (x_0, y_0), 2, 120, -1)
        # cv2.imshow('whiteDots', circle)
        # # cv2.imshow('largest contour', hsv)
        # # cv2.imwrite("/home/tim/Dokumente/poster/crob_image_hsv.png", hsv)
        # # cv2.imwrite("/home/tim/Dokumente/poster/crob_image_dots.png", circle)
        # cv2.waitKey(1)


        # cv2.waitKey()
        if not self.first_done:
            self.first_run_pos = [
                x_0, y_0,
                x_1, y_1]
            self.first_done = True
        cansee = True

        return x_0 / self.resize_image_perf + x_crop, y_0 / self.resize_image_perf + y_crop, x_1 / self.resize_image_perf + x_crop, y_1 / self.resize_image_perf + y_crop, cansee
