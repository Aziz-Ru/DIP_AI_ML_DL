import cv2
import numpy as np

# Load YOLO (use YOLOv3 trained for face detection)
net = cv2.dnn.readNetFromDarknet("yolov3-face.cfg", "yolov3-wider_16000.weights")

# Use GPU if available
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Forward pass
    detections = net.forward(output_layers)

    boxes, confidences = [], []

    # Loop over detections
    for out in detections:
        for detection in out:
            scores = detection[5:]   # class scores
            confidence = scores[0]   # since only 1 class (face)
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
