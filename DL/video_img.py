import cv2
import os

video_path = "/home/silicon/dip_ai/DL/banana.mp4"
if not os.path.isfile(video_path):
    print("Video file not found:", video_path)
    exit(1)

output_folder = "/home/silicon/Desktop/banana/"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

fps = int(cap.get(cv2.CAP_PROP_FPS))
skip = fps // 10   # 30 FPS -> save every 3rd frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % skip == 0:
        frame_filename = os.path.join(
            output_folder,
            f"frame_{saved_count:04d}.jpg"
        )
        # resize frame to 224x224
        frame = cv2.resize(frame, (224, 224))
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print("Saved:", saved_count)