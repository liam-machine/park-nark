from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from scipy.spatial import distance

# Load polygon coordinates from JSON file
with open('polygon_coords.json', 'r') as f:
    carParks = [json.loads(line) for line in f]

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "./videos/park-1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter object
output_video_path = './output/output-objects.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])

def find_closest_carpark(car_park_centroids, car_centroid, search_radius=100):
    """
    Find the closest car park centroid to a given car centroid within a search radius.
    
    Args:
    car_park_centroids (list of tuples): List of (x, y) coordinates of car park centroids.
    car_centroid (tuple): (x, y) coordinates of the car centroid.
    search_radius (int): The radius within which to search for the closest centroid.
    
    Returns:
    tuple: The closest car park centroid to the car centroid.
    """
    cx, cy = car_centroid
    closest_centroid = None
    min_distance = float('inf')
    
    for car_park_centroid in car_park_centroids:
        car_park_cx, car_park_cy = car_park_centroid
        if abs(car_park_cx - cx) <= search_radius and abs(car_park_cy - cy) <= search_radius:
            dist = distance.euclidean(car_centroid, car_park_centroid)
            if dist < min_distance:
                min_distance = dist
                closest_centroid = car_park_centroid
    
    return closest_centroid

# Loop through the video frames
prev_time = time.time()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.25)  # Set confidence threshold to 0.25

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # List to store car centroids
        car_centroids = []

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Calculate and draw the centroid of the detected object
            cx = int(x)
            cy = int(y)
            car_centroids.append((cx, cy))
            cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), -1)

        # Calculate and display the real-time frame rate
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # List to store car park centroids
        car_park_centroids = []

        # Overlay polygons and centroids on the frame
        for carPark in carParks:
            points = np.array(carPark['points'], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=True, color=(0, 255, 0), thickness=1)

            # Calculate centroid
            M = cv2.moments(points)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                car_park_centroid = (cx, cy)
                car_park_centroids.append(car_park_centroid)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(annotated_frame, f"ID: {carPark['polygon_id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw lines from car centroids to the closest car park centroid within the search radius
        for car_centroid in car_centroids:
            closest_car_park_centroid = find_closest_carpark(car_park_centroids, car_centroid, search_radius=100)
            if closest_car_park_centroid:
                cv2.line(annotated_frame, car_centroid, closest_car_park_centroid, (255, 255, 0), 2)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to", output_video_path)