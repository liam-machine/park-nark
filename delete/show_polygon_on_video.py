import cv2
import numpy as np

# Function to calculate centroid of a polygon
def calculate_centroid(polygon):
    M = cv2.moments(np.array(polygon))
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

# Function to read polygon coordinates from a file
def read_polygon_coords(file_path):
    polygons = {}
    with open(file_path, 'r') as file:
        polygon_name = None
        for line in file:
            line = line.strip()
            if line.startswith("Polygon"):
                polygon_name = line
                polygons[polygon_name] = []
            elif line and polygon_name:
                point = tuple(map(int, line.split(',')))  # Convert the coordinate to (x, y) tuple
                polygons[polygon_name].append(point)
    return polygons

# File containing polygon coordinates
polygon_file = 'polygon_coords.txt'

# Read polygon coordinates from the file
polygon_coords = read_polygon_coords(polygon_file)

# Open the video file
video_path = './videos/park-1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the output video writer
output_path = './videos/park-1_with_polygons.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw each polygon and its centroid on the frame
    for name, polygon in polygon_coords.items():
        # Draw the polygon
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Calculate and draw the centroid
        centroid = calculate_centroid(polygon)
        cv2.circle(frame, centroid, radius=5, color=(0, 0, 255), thickness=-1)
    
    # Write the frame with the drawn polygons and centroids
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
