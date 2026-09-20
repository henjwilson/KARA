import cv2
import numpy as np

# for drawing boxes
from PIL import Image

from util import get_limits


# BGR Colors: (must be in BGR)
orange = [0, 165, 255]
yellow = [0, 255, 255]
red = [0, 0, 255]
blue = [255, 0, 0]

# for video Capture. (webcam number) 0 if you only have 1 (default)
cap = cv2.VideoCapture(0)

# Exposure Settings
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # '0' typically means Manual Mode in V4L2/Linux (3 is Auto)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # apply manual Exposure

#filter out tiny noise blobs
MIN_AREA = 500

# testing Resolution: (want the widest)
resolutions_to_test = [(640, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]

# IMPORTANT: we only care about the widest resolution
max_width = 640
for w, h in resolutions_to_test:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested {w}x{h}, got {actual_w}x{actual_h}")
    if actual_w > max_width:
        max_width = actual_w
        max_height = actual_h
print(f"Best resolution found: {max_width}x{max_height}")

# set to the widest:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)

while(True):
    ret, frame = cap.read() # read from camera
    # ---------- Color Detection ----------

    # convert input image from VGR color space to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Specify color
    lowerLimit, upperLimit = get_limits(color=blue)

    # returns all the pixels of the color we want
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # cv2.imshow('mask', mask)
    # ---------- End Color Detection ----------

    # ---------- Draw Bounding Boxes (per separate blob) ----------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue  # skip small noise

        x, y, w, h = cv2.boundingRect(contour)
        # draw green rectangle around Object
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # Calculate center point
        center_x = x + w // 2
        center_y = y + h // 2
        # Draw a dot at the center
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        # Display the coordinates as text near the dot
        coord_text = f"({center_x}, {center_y})"
        cv2.putText(frame, coord_text, (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # ---------- End Draw Bounding Boxes ----------

    # Show the frame
    cv2.imshow('frame', frame)

    # press q to quit
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()

cv2.destroyAllWindows()