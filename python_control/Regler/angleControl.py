# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:43:52 2018

@author: Felix
"""
import numpy as np

def angleControl(Point_left,Point_right,u0):
    '''
    PARAMETERS
    ----------
    Point_left : nparray 2-Dim
        Point_left[0] x
        Point_left[1] y
    
    Point_right : nparray 2-Dim
        Point_right[0] x
        Point_right[1] y

    u0 : double
        u0 midline of intrinsic matrix
    '''
    Threshold=u0/3
    MaxSteering=29
    Midpoint=(Point_right+Point_left)/2
    HorizontalError=(u0-Midpoint[0])/Threshold
    if np.abs(HorizontalError)<1:
        k=1
        # possible other control schemes
        return -k*HorizontalError*MaxSteering
    else:
        return -np.sign(HorizontalError)*MaxSteering
