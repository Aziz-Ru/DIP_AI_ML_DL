import cv2
import numpy as np

img = np.zeros((200, 200), dtype=np.uint8)

cv2.putText(img, "A", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,4, (255), thickness=8)

cv2.imwrite("binary_A.png", img)


# cv2.imshow("Binary A", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
