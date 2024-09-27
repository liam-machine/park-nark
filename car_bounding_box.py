import cv2

# Load pre-trained car classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

# Open video capture (0 for webcam, or you can pass a video file path)
cap = cv2.VideoCapture('./videos/park-1.mp4')  # Replace with 0 if you want to use webcam

while cap.isOpened():
    # Read frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break  # Break when the video ends or there's an issue with the frame

    # Convert to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(gray_frame, 1.1, 1)

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Car Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
