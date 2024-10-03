import cv2
import json
import numpy as np

# Load polygon coordinates from JSON file
with open('polygon_coords.json', 'r') as f:
    polygons = [json.loads(line) for line in f]

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Overlay polygons and centroids on the frame
    for polygon in polygons:
        points = np.array(polygon['points'], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Calculate centroid
        M = cv2.moments(points)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {polygon['polygon_id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to", output_video_path)