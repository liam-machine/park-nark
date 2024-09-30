import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch 
from deep_sort_realtime.deepsort_tracker import DeepSort 

# Check if a GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model with the correct device
model = YOLO("yolov8s.pt").to(device)
# Define the video source
source = './videos/park-1.mp4'
cap = cv2.VideoCapture(source)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Reduce video frame size for faster processing
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the points of the first polygon (example coordinates)
x1 = 425
y1 = 720
width1 = 140
height1 = 40
points1 = np.array([[x1, y1], [(x1 + width1), y1], [(x1 + width1 - 40), (y1 - height1)], [x1 - 20, (y1 - height1 + 10)]], np.float32)

# Define the points of the second polygon (example coordinates)
x2 = 540
y2 = 480
width2 = 60
height2 = 25
points2 = np.array([[x2, y2], [(x2 + width2), y2-5], [(x2 + width2 - 33), (y2 - height2)], [(x2 -33), (y2 - height2)]], np.float32)

# Function to check if a bounding box is inside a polygon
def is_box_in_polygon(box, polygon):
    x1, y1, x2, y2 = box
    bbox_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return any(cv2.pointPolygonTest(polygon, tuple(point), False) >= 0 for point in bbox_points)

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

    # Initialize polygon colors
    color1 = (0, 255, 0)  # Green for the first polygon
    color2 = (0, 255, 0)  # Green for the second polygon

    # Check if any bounding boxes intersect with the polygons
    for result in results:
        for box in result.boxes:
            x1_box, y1_box, x2_box, y2_box = box.xyxy[0].numpy()  # Bounding box coordinates

            # Check for polygon 1
            if is_box_in_polygon((x1_box, y1_box, x2_box, y2_box), points1):
                color1 = (0, 0, 255)  # Red if the bounding box is inside polygon 1

            # Check for polygon 2
            if is_box_in_polygon((x1_box, y1_box, x2_box, y2_box), points2):
                color2 = (0, 0, 255)  # Red if the bounding box is inside polygon 2

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(x1_box), int(y1_box)), (int(x2_box), int(y2_box)), (0, 255, 0), 2)
            label = model.names[int(box.cls[0])]  # Class label
            cv2.putText(frame, f'{label}', (int(x1_box), int(y1_box) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the polygons with the updated colors
    polygon_points1 = points1.reshape((-1, 1, 2))  # Reshape for polylines
    cv2.polylines(frame, [polygon_points1.astype(np.int32)], isClosed=True, color=color1, thickness=2)

    polygon_points2 = points2.reshape((-1, 1, 2))  # Reshape for polylines
    cv2.polylines(frame, [polygon_points2.astype(np.int32)], isClosed=True, color=color2, thickness=2)

    # Display the frame with detections and polygons
    cv2.imshow('Parking Detection', frame)

    # End timing and print FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps}")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
