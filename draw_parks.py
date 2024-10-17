import json
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
        polygon_data = {
            "polygon_id": polygon_count,
            "points": points
        }

        # Write the polygon data to the JSON file
        with open("polygon_coords.json", "a") as json_file:
            json.dump(polygon_data, json_file)
            json_file.write("\n")

        # Save the polygon to the list of polygons
        polygons.append(points.copy())

        print(f"Polygon {polygon_count} saved to JSON file and displayed.")
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


# 0 > 8m
# 1 > 13.4m
# 2 > ?
# 3 > +5.4m
# 4 > +5.4m
# 5 > ?
# 6 > +5.4m
# 7 > +5.4m

Adding point #0 with position(413,715)
Adding point #1 with position(411,667)
Adding point #2 with position(400,554)
Adding point #3 with position(397,498)
Adding point #4 with position(394,459)
Adding point #5 with position(383,422)
Adding point #6 with position(379,402)
Adding point #7 with position(374,378)
Adding point #8 with position(373,367)
Adding point #9 with position(369,353)