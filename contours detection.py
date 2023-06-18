import cv2 as cv

img = cv.imread(r"D:\NUST\2nd Semester\computer Vision\fish blob.png")
# img = cv.bitwise_not(img)
edge = cv.Canny(img, 50, 200)

contours, heirarchy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(img, contours, -1, (0, 255, 0), 2)


cv.imshow('Edges', edge)
cv.imshow('Contours', img)
cv.waitKey()