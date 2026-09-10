import cv2 as cv
import numpy as np

img=cv.imread(r'C:/Coding/ML_DL/OpenCV/boston.jpg')
cv.imshow('bos',img)

#translation
def translate(img,x,y):
    transMatrix=np.float32([[1,0,x],[0,1,y]])
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMatrix,dimensions)

translated=translate(img,100,100)
# cv.imshow('translated',translated)

#rotation
def rotate(img,angle,rotPoint=None):
    height,width=img.shape[:2]

    if rotPoint is None:
        rotPoint=(width//2,height//2)
        
    rotMatrix=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(width,height)

    return cv.warpAffine(img,rotMatrix,dimensions)

rotated=rotate(img,90)
# cv.imshow('rotated',rotated)

# flipping

flipped=cv.flip(img,0)
cv.imshow('flipped',flipped)


cv.waitKey(0)