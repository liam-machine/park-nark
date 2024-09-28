import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Define the video source
source = './videos/park-1.mp4'
cap = cv2.VideoCapture(source)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through video frames
while True:
    ret, frame = cap.read()
    
    # Break the loop if no more frames
    if not ret:
        break
    
    # Use YOLO model to predict objects in the frame
    results = model.predict(source=frame, save=False, save_txt=False)
    
    # Get detection data from results and plot them on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('YOLO Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
