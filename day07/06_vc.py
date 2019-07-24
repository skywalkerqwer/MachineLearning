import cv2 as cv
vc = cv.VideoCapture(0)
while True:
    frame = vc.read()[1]
    cv.imshow('frame', frame)
    if cv.waitKey(33) == 27:
        break
vc.release()
cv.destroyAllWindows()
