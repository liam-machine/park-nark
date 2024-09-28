import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch 

# Check if a GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model with the correct device
model = YOLO("yolov8s.pt").to(device)

# Initialize DeepSORT without iou_threshold parameter
deepsort = DeepSort(max_age=30)

# Define the video source
source = './videos/park-1.mp4'
cap = cv2.VideoCapture(source)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set class filter (e.g., detect only 'car' and 'truck' for parking lot)
class_filter = ['car', 'truck']

# Convert class names to their corresponding IDs from the model.names dictionary
class_ids = [cls_id for cls_id, cls_name in model.names.items() if cls_name in class_filter]

# Loop through video frames
frame_count = 0
frame_skip = 3  # Process every 3rd frame

while True:
    ret, frame = cap.read()

    # Break the loop if no more frames
    if not ret:
        break

    # Skip frames to reduce load
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Start timing
    start_time = time.time()

    # Use YOLO model to predict objects in the frame
    results = model.predict(source=frame, save=False, save_txt=False, classes=class_ids)

    # Prepare detections for DeepSORT
   # Prepare detections for DeepSORT
    # Prepare detections for DeepSORT
    # Prepare detections for DeepSORT
    detections = []
    for result in results:
        for box in result.boxes:
            x1_box, y1_box, x2_box, y2_box = box.xyxy[0].numpy()  # Bounding box coordinates
            confidence = box.conf[0].item()  # Get confidence score
            cls_id = int(box.cls[0])  # Class ID

            # Collect only car and truck detections
            if cls_id in class_ids:
                # Append detection as a list
                detections.append([
                    float(x1_box), 
                    float(y1_box), 
                    float(x2_box), 
                    float(y2_box), 
                    float(confidence)
                ])

    # Debugging: Print the detections to verify their format
    print("Detections:", detections)

    # Update DeepSORT tracker
    if detections:  # Ensure that there are detections
        tracks = deepsort.update_tracks(detections, frame=frame)
    else:
        tracks = []  # If no detections, initialize tracks as empty



    # Draw tracks and IDs
    for track in tracks:
        x1, y1, x2, y2, track_id = track.to_xyah()  # Convert to xyah format
        # Draw the bounding box on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put the ID label
        cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections and IDs
    cv2.imshow('Parking Detection', frame)

    # End timing and print FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
