# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:18:11 2021

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
        # Find the chess board corners
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
images = glob.glob("./input"+'/' + "Image" + '*.' + "tif")

sift = cv2.xfeatures2d.SIFT_create(1000)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 

img1 = cv2.imread("./input/Image1.tif",0) 
kp1, des1 = sift.detectAndCompute(img1,None)

solution_mats = []
for fname in images[1:]:
    
    img2 = cv2.imread(fname,0) 

    kp2, des2 = sift.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    
    kps1 = np.float32([kp.pt for kp in kp1])
    kps2 = np.float32([kp.pt for kp in kp2])
    
    pts1 = np.float32([kps1[m.queryIdx] for m in matches])
    pts2 = np.float32([kps2[m.trainIdx] for m in matches])
            
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    H, status = cv2.findHomography(pts1, pts2)
    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H, mtx)
    solution_mats.append([num, Rs, Ts, Ns])
   