import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 nano model (smaller and faster)
model = YOLO("yolov8n.pt")

# Define the video source
source = './videos/park-1.mp4'
cap = cv2.VideoCapture(source)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Desired frame size
frame_width = 640
frame_height = 480

# Loop through video frames
while True:
    ret, frame = cap.read()

    # Break the loop if no more frames
    if not ret:
        break

    # Resize the frame explicitly
    resized_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

    # Display the resized frame in a window
    cv2.imshow('Resized Video', resized_frame)

    # Add a small delay to simulate real-time display
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
