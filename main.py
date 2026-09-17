import cv2

# for drawing boxes
from PIL import Image

#
from util import get_limits


yellow = [0, 255, 255]  # yellow in RGB color space
red = [255, 0, 0]

# for video Capture. (webcam number) 0 if you only have 1 (default)
cap = cv2.VideoCapture(1)

# Set manual exposure (value range depends on your specific camera driver)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # '0' typically means Manual Mode in V4L2/Linux (3 is Auto)
# If on Windows/macOS and '1' does not work, try passing '0' or '0.25' instead.
# 2. Now apply your manual exposure value
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

MIN_AREA = 500  #filter out tiny noise blobs

while(True):
    ret, frame = cap.read() # read from camera

    # ---------- Color Detection ----------
        # Convert VGR color space to RGB:
    # convert input image from VGR color space to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Specify Yellow
    lowerLimit, upperLimit = get_limits(color=yellow)

    # returns all the pixels of the color we want
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # ---------- End Color Detection ----------

    # ---------- Draw Bounding Boxes (per separate blob) ----------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue  # skip small noise

        x, y, w, h = cv2.boundingRect(contour)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    # ---------- End Draw Bounding Boxes ----------

    # converting immage from opencv array to pill array
    # mask_ = Image.fromarray(mask)
    # get the bounding box
    # bbox = mask_.getbbox()

    # if bbox is not None:
        # x1, y1, x2, y2, = bbox

        # draw rectangle: frame, (top corner), (bottom corner), color (green), line thickness
        ## frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # print(bbox)
    # ---------- Draw Bounding Box ----------

    # Show the frame
    cv2.imshow('frame', frame)

    # press q to quit
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()

cv2.destroyAllWindows()