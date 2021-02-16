# Camera-Calibration
This repo contains examples of estimating distortion parameters and unidistortion, estimating intrinsic, projection, essential matrixes and epipolar lines.

a)
In this part, the program finds the intrinsic parameters and distortion parameters by using a function
called calibrate. This function calibrates the camera and finds its parameters by using the chessboard
technique. It takes some inputs, image paths, image extensions, the number of chess squares in the
image, and the size of one edge of a chess square. I have twenty images that contain chess boards
but the “findChessboardCorners” method did not use all of it because it did not find corners properly
in some images. I filtered images before giving them to the function and I increased the useful image
number to thirteen. After finding the corners, the program increases the accuracy of these corner
points by using the “cornerSubPix” method. Then program calls the “calibrateCamera” function using
all the information. With this, we will estimate the intrinsic parameters and distortion parameters.
Then using distortion parameters and camera matrix, the program undistorts all images. You can see
the results in the output folder.

b)
In this part, program estimates the intrinsic parameters using the above method.

c)
For this part, we will estimate the projection matrix. We already have extrinsic and intrinsic
parameters for 13 images but the “calibrateCamera” method returns a rotation matrix in the shape
of 3x1. We need to 3x3 rotation matrix for projection matrix calculation. So, we need to convert it
3x3 by using the “Rodriges” function. After that, the program calculates the projection matrixes for
the images.


d)
In this part, the program calculates the essential matrixes for twenty images by using SIFT detector
key points and camera matrix. First, the program calculates the camera matrix again then it extracts
the keypoints by using SIFT, then picks the best ones. After these steps, it calculates essential
matrixes between the first image and other images by using the “findEssentialMat” function.

e)
In this part, we will find the translation and rotation matrixes between the first image and the
remaining images. In order to do that, first, we will calculate the homography matrix with the SIFT
extractor. Then we need to calculate the camera matrix for decomposition. By using the calibrate
method again we will calculate the camera matrix. By using the “decomposeHomographyMat”
function we will decompose the homography matrix with the camera matrix. In the end, we will have
rotation and translation matrixes. For each image, we may have more than one rotation and
translation matrixes. You may find all of them in the solution matrix.

f)
In the last part, we will draw some epipolar lines between the two images. To achieve this, we will
use again the camera matrix and key point matches between two images. I already mentioned the
calculation of the camera matrix and finding matching and selecting the best key point matches
between images. So, by using these we will calculate the fundamental matrix. This matrix helps us
when we draw epipolar lines on the images. The “computeCorrespondEpilines” estimates the
epipolar lines by using key points and fundamental matrix then the “drawLines” method draws the
lines onto the image. You can find the result in the output folder.
