# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:39:15 2021

@author: Abdullah Hamza Şahin
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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


def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        
    return img1



ret, mtx, dist, rvecs, tvecs = calibrate(".\input", "Image", "tif", 0.3, 12, 12)


sift = cv2.xfeatures2d.SIFT_create(1000)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
img1 = cv2.imread("./input/Image13.tif",0) 

kp1, des1 = sift.detectAndCompute(img1,None)


img2 = cv2.imread("./input/Image14.tif",0) 

kp2, des2 = sift.detectAndCompute(img2,None)
matches = bf.match(des1,des2)

kps1 = np.float32([kp.pt for kp in kp1])
kps2 = np.float32([kp.pt for kp in kp2])

pts1 = np.float32([kps1[m.queryIdx] for m in matches])
pts2 = np.float32([kps2[m.trainIdx] for m in matches])
        
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img1_draw = drawlines(img1,img2,lines1,pts1,pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img2_draw = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img1_draw)
plt.subplot(122),plt.imshow(img2_draw)
plt.savefig("./output/Epipolar_Lines.png")
plt.show()