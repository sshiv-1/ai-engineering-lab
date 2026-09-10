import cv2 as cv

capture = cv.VideoCapture(0)

#resizing and rescaling
def rescale(frame,scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)

while True:
    ret, frame = capture.read()
    frame_resized=rescale(frame)
    if not ret:
        print("Failed to grab frame")
        break

    cv.imshow('Frame', frame)
    cv.imshow('Resized Frame', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break



capture.release()
cv.destroyAllWindows()
