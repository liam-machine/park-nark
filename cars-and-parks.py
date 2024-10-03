import cv2
import json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Load polygon coordinates from JSON file
with open('polygon_coords.json', 'r') as f:
    polygons = [json.loads(line) for line in f]

# Function to calculate the centroid of a polygon
def calculate_centroid(points):
    M = cv2.moments(points)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    return None

# Open the input video file
input_video_path = './videos/park-1.mp4'
output_video_path = './output/output_with_parks.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter object
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Annotate the frame with tracking lines and polygons
    annotated_frame = frame.copy()
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 30 tracks for 30 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    # Overlay polygons and their centroids
    for polygon in polygons:
        points = np.array(polygon['points'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        centroid = calculate_centroid(points)
        if centroid:
            cx, cy = centroid
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"ID: {polygon['polygon_id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("YOLO Tracking with Polygons", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to", output_video_path)