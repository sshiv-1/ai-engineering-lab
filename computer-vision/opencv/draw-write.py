import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
blank[:] = 0, 255,0  # fills entire image with green
cv.imshow('blank',blank)
# img = cv.imread(r'C:/Coding/ML_DL/OpenCV/cat.jpg')
# cv.imshow('cat',img)

#draw a rectange
# cv.rectangle(blank,(0,0),(250,2500),(255,0,0),thickness=cv.FILLED)
# cv.imshow('blank',blank)

#circle
cv.circle(blank,(250,250),40,(0,0,255),thickness=cv.FILLED)
cv.imshow("circle",blank)
cv.waitKey(0)