# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:24:16 2021

@author: Abdullah Hamza Şahin
"""

import numpy as np
import cv2
import glob



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, prefix, image_format, square_size, width=12, height=12):
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    objpoints = []  
    imgpoints = [] 

    

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = cv2.GaussianBlur(gray,(3,3),0)
        sharpeningKernel = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")
        out = cv2.filter2D(out, -1, sharpeningKernel)
        
        ret, corners = cv2.findChessboardCorners(out, (width, height), flags = cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        
        print(ret)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]





ret, mtx, dist, rvecs, tvecs = calibrate(".\input", "Image", "tif", 0.3, 12, 12)

projection_matrixs = []
for i in range(len(rvecs)):
    rotation_mat = np.zeros(shape=(3, 3))
    R = cv2.Rodrigues(rvecs[i], rotation_mat)[0]
    P = np.column_stack((np.matmul(mtx,R), tvecs[i]))
    projection_matrixs.append(P)
