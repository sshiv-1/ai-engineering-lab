import cv2 as cv
img=cv.imread(r'C:/Coding/ML_DL/OpenCV/boston.jpg')
cv.imshow('bos',img)

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

# blur=cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
# cv.imshow('blur',blur)


#edge cascade
# canny=cv.Canny(img,125,175)
# cv.imshow('canny',canny)

# #dialating the images
# dilated=cv.dilate(canny,(8,8),iterations=1)
# cv.imshow('dialated',dilated)
# #eroding
# eroded=cv.erode(dilated,(3,3),iterations=1)
# cv.imshow('eroded',eroded)

#resize
resized= cv.resize(img,(500,500),interpolation=cv.INTER_LINEAR)
cv.imshow('resized',resized)

cropped=img[50:200,200:400]
cv.imshow('croppedt',cropped)

cv.waitKey(0)
