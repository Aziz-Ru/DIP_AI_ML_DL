import cv2 as cv
import numpy as np



# cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# -images: A list of image arrays (usually just [image]).Must be passed in square brackets.
#  channels: Indexes of the channels for which the histogram is calculated.
# For a grayscale image: [0]For a color image (e.g., BGR):Blue: [0], Green: [1], Red: [2]
#  You can also compute multi-dimensional histograms like channels=[0,1]
# mask:Optional mask.If you want to compute the histogram for the whole image, pass None
# histSize:Number of bins for each channel.Must be a list or tuple.
# Example: [256] for grayscale or per-channel bins for color histograms like [8, 8

# ranges:The range of pixel values.Usually [0, 256] for 8-bit images (0 to 255).

def draw_histogram(image,color):
  hist=cv.calcHist([image],[0],None,[256],[0,256])
  hist=cv.normalize(hist,hist).flatten()
  hist_img=np.zeros((200,256,3),dtype=np.uint8)

  for x in range(256):
    val = int(hist[x]*200)
    if color=='b':
      cv.line(hist_img,(x,200),(x,200-val),(255,0,0))
    elif color=='g':
      cv.line(hist_img,(x,200),(x,200-val),(0,255,0))
    elif color=='r':
      cv.line(hist_img,(x,200),(x,200-val),(0,0,255))
  return hist_img 


def show_camera():
  cap=cv.VideoCapture(0)

  while True:
    _,frame=cap.read()
    if not _:
      break

    b,g,r=cv.split(frame)
    hist_b=draw_histogram(b,'b')
    hist_g=draw_histogram(g,'g')
    hist_r=draw_histogram(r,'r')

    cv.imshow("Live Camera",frame)
    cv.imshow("Blue Channel Histogram",hist_b)
    cv.imshow("Green Channel Histogram",hist_g)
    cv.imshow("Red Channel Histogram",hist_r)

    # cv.imshow("Red Channel",r,)
    # cv.imshow("Green Channel",g,)
    # cv.imshow("Blue Channel",b,)
    
     # Wait 1 ms for key press; exit on 'q'
    if cv.waitKey(1) & 0xFF==ord('q'):
      break
  
  cap.release()
  cv.destroyAllWindows()


if __name__=='__main__':
  show_camera()