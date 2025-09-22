import cv2 as cv

# reading images
# img= cv.imread('./image/image-3.jpeg')
# cv.imshow('RGB',img)

# reading video
capture=cv.VideoCapture(0)
while True:
  isTrue,frame=capture.read()
  cv.imshow("Video",frame)

  if cv.waitKey(20) & 0xFF== ord('d'):
    break

capture.release()
cv.destroyAllWindows()
