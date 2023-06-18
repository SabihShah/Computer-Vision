import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(f"D:\OpenCV\wall.jpg")
# cv.imshow("img", img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier(r"D:\NUST\2nd Semester\computer Vision\Face detection (Haar_Cascade)\faces.xml")

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
print(face_rect)

print("Faces found:", len(face_rect))

for x,y,w,h in face_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
cv.imshow("Detected Faces", img)
cv.waitKey()