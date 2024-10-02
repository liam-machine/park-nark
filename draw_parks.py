import cv2
import numpy as np

# ***** replace with required image path *****
path = "first_frame.jpg"
img = cv2.imread(path)
clone = img.copy()
temp = img.copy()

# ***** global variable decleration *****
done = False
points = []
current = (0, 0)
prev_current = (0, 0)
polygon_count = 0
polygons = []  # List to store all polygons

# Open a file to save the polygon coordinates
file = open("polygon_coords.txt", "w")

def on_mouse(event, x, y, buttons, user_param):
    global done, points, current, temp
    # Mouse callback for drawing polygons
    if done:
        return
    if event == cv2.EVENT_MOUSEMOVE:
        current = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click to add points
        print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
        cv2.circle(img, (x, y), 5, (0, 200, 0), -1)
        points.append([x, y])
        temp = img.copy()
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to finish polygon
        print("Completing polygon with %d points." % len(points))
        done = True

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_mouse)

while True:
    # Reset the image to the original clone
    img = clone.copy()

    # Draw all the previous polygons
    for polygon in polygons:
        cv2.polylines(img, [np.array(polygon)], True, (255, 0, 0), 1)

    # Draw the current polygon outline if there are points
    if len(points) > 1:
        cv2.polylines(img, [np.array(points)], False, (255, 0, 0), 1)
        cv2.line(img, (points[-1][0], points[-1][1]), current, (0, 0, 255))

    # Update the window
    cv2.imshow("image", img)
    key = cv2.waitKey(50)

    # Save the polygon when 'd' is pressed
    if key == ord('d') and len(points) > 2:
        # Save the polygon coordinates to file
        file.write(f"Polygon {polygon_count}:\n")
        for point in points:
            file.write(f"{point[0]},{point[1]}\n")
        file.write("\n")

        # Save the polygon to the list of polygons
        polygons.append(points.copy())

        print(f"Polygon {polygon_count} saved to file and displayed.")
        polygon_count += 1
        points = []  # Reset points to allow drawing a new polygon
        done = False

    # Exit and close the file when 'q' is pressed
    if key == ord('q'):
        print("Closing the file and exiting.")
        break

# Close the file and release resources
file.close()
cv2.destroyAllWindows()
