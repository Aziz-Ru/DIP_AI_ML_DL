# face_detect_webcam.py
import cv2 as cv

def main():
    # Use OpenCV's built-in Haar cascade (path is bundled with cv2)
    cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade. Check your OpenCV install.")

    cap = cv.VideoCapture(0)  # change to 1/2 if you have multiple cameras
    if not cap.isOpened():
        raise RuntimeError("Could not access the camera.")

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, exiting.")
            break

        # Convert to grayscale for the detector
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces (tune minNeighbors/scaleFactor for your setup)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # image pyramid step
            minNeighbors=5,       # higher → fewer false positives
            minSize=(60, 60)      # ignore very small detections
        )

        # Draw detections
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, "Face", (x, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

        # Show count
        cv.putText(frame, f"Faces: {len(faces)}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Webcam Face Detection (Haar Cascade)", frame)

        # Quit with 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
