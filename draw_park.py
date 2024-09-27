import cv2
import numpy as np

# Define the points of the first polygon (example coordinates)
x1 = 410
y1 = 720
width1 = 175
height1 = 60
points1 = np.array([[x1, y1], [(x1 + width1), y1], [(x1 + width1 - 40), (y1 - height1)], [x1 - 20, (y1 - height1 + 10)]], np.int32)

# Define the points of the second polygon (example coordinates)
x2 = 523
y2 = 490
width2 = 100
height2 = 35
points2 = np.array([[x2, y2], [(x2 + width2), y2-5], [(x2 + width2 - 49), (y2 - height2)], [(x2 -33), (y2 - height2 )]], np.int32)

# Open the video file
input_video_path = './videos/park-1.mp4'  # Replace with your input video path
cap = cv2.VideoCapture(input_video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if there are no more frames

    # Draw the first polygon on the frame
    polygon_points1 = points1.reshape((-1, 1, 2))  # Reshape for polylines
    cv2.polylines(frame, [polygon_points1], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw the second polygon on the frame
    polygon_points2 = points2.reshape((-1, 1, 2))  # Reshape for polylines
    cv2.polylines(frame, [polygon_points2], isClosed=True, color=(0, 255, 0), thickness=2)  # Different color for visibility

    # Display the modified frame
    cv2.imshow('Video with Bounding Polygons', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
