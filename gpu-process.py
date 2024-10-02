import datetime
from ultralytics import YOLO
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CAR_CLASS_ID = 2

output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_cap = cv2.VideoCapture("./videos/park-1.mp4")
video_writer = create_video_writer(video_cap, os.path.join(output_dir, "output.mp4"))

# Initialize YOLO model and DeepSort tracker
model = YOLO("yolov5s.pt")
tracker = DeepSort(max_age=30, n_init=3)

while video_cap.isOpened():
    ret, frame = video_cap.read()
    if not ret:
        break

    # Process frame with YOLO model
    results = model(frame)

    for result in results:
        print(result.names)
        print(result.
        print(result.boxes.xyxy[0])
        print(result.probs)

    # print(results)
    # Extract bounding boxes, confidence scores, and class IDs
    detections = []
    # for result in results.xyxy[0]:  # Assuming results.xyxy[0] contains the detections
        # bbox = result[:4].tolist()  # Bounding box coordinates
        # confidence = result[4].item()  # Confidence score
        # class_id = int(result[5].item())  # Class ID

        # if confidence > CONFIDENCE_THRESHOLD:
        #     detections.append({
        #         'bbox': bbox,
        #         'confidence': confidence,
        #         'class_id': class_id
        #     })

    # Update tracker with detections
    tracker.update(detections)

    # Draw bounding boxes and labels on the frame
    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class_id']
        if class_id == CAR_CLASS_ID:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), GREEN, 2)
            cv2.putText(frame, "Car", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

    # Write the processed frame to the output video
    video_writer.write(frame)

    # Release GPU memory
    torch.cuda.empty_cache()

# Release resources
video_cap.release()
video_writer.release()
cv2.destroyAllWindows()