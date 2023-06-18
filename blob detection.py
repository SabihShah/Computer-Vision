import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread(r"D:\NUST\2nd Semester\computer Vision\red.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = cv.bitwise_not(gray)

params = cv.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 200
params.filterByArea = True
params.minArea = 10
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv.SimpleBlobDetector_create(params)

keypoints = detector.detect(gray)
print(keypoints)

image_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0),
                                         cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('BLOBS', image_with_keypoints)
cv.waitKey(0)