import cv2 as cv
import numpy as np

img=cv.imread(r'C:/Coding/ML_DL/OpenCV/boston.jpg')
cv.imshow('bos',img)

#bgr to gray
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)


#bgr to hsv
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow('hsv',hsv)

#lab
lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow('lab',lab)


cv.waitKey(0)