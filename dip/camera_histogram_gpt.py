import cv2
import numpy as np

def show_camera_histogram():
    cap = cv2.VideoCapture(0)  # 0 = default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Normalize histogram to fit display height
        hist = cv2.normalize(hist, hist).flatten()

        # Create black canvas to draw histogram
        canvas = np.zeros((300, 256), dtype=np.uint8)

        for x in range(256):
            cv2.line(canvas, (x, 300), (x, 300 - int(hist[x] * 300)), 255)

        # Show webcam and histogram
        cv2.imshow('Webcam', gray)
        cv2.imshow('Histogram', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

show_camera_histogram()
