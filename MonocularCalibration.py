#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
### 3rd-party dependency:
sudo pip install opencv-python
'''

import numpy as np
import cv2, sys, time, json, sys, os, glob

input_images = "./1280_720/*.jpg"
# chessboard inner corners, (width, height)
chessboard = (29, 19) #(21, 14)
# chessboard square side lenght, mm
square_side_lenght = 1 #chessboard square side lenght not necessary in monocular calibration

class MonocularCalibration:
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    objpoints = []
    imgpoints = []
    images = []
    cameraMatrix = []
    distCoeffs = []

    def __init__(self):
        print("MonocularCalibration init")
        self.objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
        self.objp *= square_side_lenght
        #print("chessboard points: ", self.objp)

    def __del__(self):
        print("MonocularCalibration quit")

    def LoadImages(self, inputs):
        self.images = sorted(glob.glob(inputs))
        print( "found {} images: {}".format(len(self.images), self.images) )

    def GetImageByIndex(self, index):
        return self.images[index]

    def DoCalib(self):
        for fname in self.images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard, None)
            # save corners
            if ret == True:
                cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), self.CRITERIA)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                # show corners
                cv2.drawChessboardCorners(img, chessboard, corners, ret)
		cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                cv2.imshow('findCorners', img)
                cv2.waitKey(-1)
        cv2.destroyAllWindows()
        retval, self.cameraMatrix, self.distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        print("overall RMS re-projection error: {}".format(retval))
        print("Intrinsics matrix: {}".format(self.cameraMatrix))
        print("distortion coefficients (k1,k2,p1,p2,k3): {}".format(self.distCoeffs))

        total_error = 0 #less is better
        for i in xrange(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], self.cameraMatrix, self.distCoeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            total_error += error
        print("total projection error: {}".format(total_error/len(self.objpoints)))

    def UndistortImage(self, fname):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newCameraMatrix, roi=cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, (w,h), 1, (w,h))
        dst = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None, newCameraMatrix)
        cv2.imshow('undistort', dst)
        cv2.waitKey(-1)
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    os.system("reset")
    try:
        wrapper = MonocularCalibration()
        wrapper.LoadImages(input_images)
        wrapper.DoCalib()
        time.sleep(1)
        #pick the 7th image, undistort & show
        wrapper.UndistortImage(wrapper.GetImageByIndex(0))
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print('break by user.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
