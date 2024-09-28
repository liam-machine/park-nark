import cv2
import numpy as np
import torch
from yolov5 import utils
from yolov5.models.yolov5s import YOLOv5s
from deepsort.tracker import Tracker
from deepsort.detection import Detection
from deepsort.deep_sort import DeepSort

# Load YOLOv5 model
model = YOLOv5s().eval()
model.load_state_dict(torch.load('yolov5s.pt'))

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=5, min_hits=3)

# Open video file
cap = cv2.VideoCapture('./videos/park-1.mp4')

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Run YOLOv5 inference
    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()

    # Convert YOLOv5 detections to Deep SORT format
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        confidence = results.xyxy[0][:, -2][0]
        detection = Detection(box, confidence)
        detections.append(detection)

    # Update Deep SORT tracker
    tracker.update(detections)

    # Draw tracked objects
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()