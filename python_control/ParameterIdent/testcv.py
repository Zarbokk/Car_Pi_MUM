# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:20:34 2018

@author: Felix
"""

import numpy as np
import cv2
import glob
import matplotlib as plt
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# chessgrid
grid=(6,8)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid[0]*grid[1],3), np.float32)
objp[:,:2] = np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)*24.8#24.8mm is the size of the chessboards


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('computervision\\*.jpg')
images = cv2.VideoCapture("/home/tim/Documents/Car_Pi_MUM/python_control/ParameterIdent/calibration_data.avi")
ok, frame = images.read()
while ok:
    #img = cv2.imread(fname)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('largest contour', gray)
    #cv2.waitKey()
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (grid[0],grid[1]),None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
    ok, frame = images.read()
        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (grid[0],grid[1]), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(ret)
print(mtx)
print(np.asarray(np.invert(mtx)))
np.savez("camParam",ret, mtx, dist, rvecs, tvecs)
# mtx are intrinsic parameters: focallength and optical centers
# rotation and translationvectors for each chesspattern
