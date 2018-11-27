# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:24:22 2018

@author: Felix
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

with np.load("camParam.npz") as X:
    mtx, dist = [X[i] for i in ('arr_1','arr_2')]

homom=np.array([1,0,0,0,0,1,0,0,0,0,1,0]).reshape((3,4))
intrinsic=mtx.dot(homom) # homogeneous matrix of intrinsic values
# example of maping world points to imagepoints first of the normal grid in objp
# then a rotated version of these world points
objp_h=cv2.convertPointsToHomogeneous(objp)
th=30/180*np.pi
T_rt=np.array([[np.cos(th),np.sin(th),0,-5],[-np.sin(th),np.cos(th),0,0],[0,0,1,0],[0,0,10,1]]).reshape((4,4))
C=intrinsic.dot(T_rt)
imagep_r=list(map(lambda ob:C.dot(ob.T),objp_h)) 
imagep=list(map(lambda ob:intrinsic.dot(ob.T),objp_h))

plt.figure()
plt.scatter([ob[0]for ob in imagep],[ob[1]for ob in imagep])
plt.figure()
plt.scatter([ob[0]for ob in imagep_r],[ob[1]for ob in imagep_r])