from ultralytics import YOLO

model = YOLO('/home/silicon/dip_ai/weights/best.pt')

model.predict(
    source="/home/silicon/Desktop/sandel_video.mp4",
    show=True,
    conf=0.3
)