import numpy as np
import cv2


COLOR_RANGES = {
    # Wider HSV range tolerates warm/cool lighting and slightly faded yellow.
    "yellow": (((15, 80, 70), (42, 255, 255)),),
}

COLOR_BGR = {
    "yellow": (0, 255, 255),
}


def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit


# The top face must be this share of the blob to be trusted; outside it the
# brightness split found glare or shadow rather than a face.
TOP_FACE_AREA_RANGE = (0.2, 0.9)


def _rect_contact_xy(box):
    """Midpoint of the rectangle's lowest edge in the image."""
    order = np.argsort(box[:, 1])
    bottom = box[order[-2:]]
    return float(bottom[:, 0].mean()), float(bottom[:, 1].mean())


def _centroid(points):
    m = cv2.moments(points)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def _find_top_face(value, contour):
    """Rotated box of the block's top face, or None if it can't be told apart.

    Lit from above, the top face is the brightest part of the blob, so the
    blob's pixels are split by brightness (Otsu) and the largest bright
    region is taken.
    """
    x, y, w, h = cv2.boundingRect(contour)
    blob = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blob, [contour - (x, y)], -1, 255, -1)
    roi = value[y:y + h, x:x + w]
    pixels = roi[blob > 0].reshape(-1, 1)
    threshold, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = ((roi > threshold) & (blob > 0)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    faces, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not faces:
        return None
    face = max(faces, key=cv2.contourArea)
    share = cv2.contourArea(face) / max(cv2.contourArea(contour), 1.0)
    if not TOP_FACE_AREA_RANGE[0] <= share <= TOP_FACE_AREA_RANGE[1]:
        return None
    return cv2.boxPoints(cv2.minAreaRect(face + (x, y))).astype(np.float32)


def detect_colored_blocks(
    frame,
    min_area=400,
    include_mask=False,
    border=None,
    border_margin=0.0,
    min_fill=0.6,
):
    """Return colored blocks with their top-face centers and table points.

    Each detection is fitted with a rotated rectangle; blobs that fill less
    than ``min_fill`` of that rectangle are rejected as non-rectangular.
    When ``border`` (an image-space polygon) is given, only blocks whose
    center lies inside it, or within ``border_margin`` pixels outside it,
    are kept.

    ``top_xy`` is the image center of the block's top face. The calibration
    maps the table plane, so ``contact_xy`` is the table point directly
    below it: the visible blob is the top face plus its own copy shifted to
    the bottom face, so the blob's center lies halfway between the top and
    bottom centers. When no top face is found, the blob's lowest edge
    midpoint is used instead (``top_found`` False).
    """
    # Blur first so single-pixel sensor noise doesn't cross the threshold.
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = np.ones((3, 3), dtype=np.uint8)
    detections = []
    if border is not None:
        border = np.asarray(border, dtype=np.float32).reshape(-1, 1, 2)

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), _angle = rect
            # Ignore very thin regions that are usually color noise or edges.
            if min(width, height) < 8:
                continue
            if area / (width * height) < min_fill:
                continue
            if border is not None and cv2.pointPolygonTest(
                border, (float(cx), float(cy)), True
            ) < -border_margin:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            found.append((cx, cy, box, contour))
        found.sort(key=lambda item: (item[0], item[1]))
        for index, (cx, cy, box, contour) in enumerate(found, start=1):
            left, top = np.floor(box.min(axis=0)).astype(int)
            right, bottom = np.ceil(box.max(axis=0)).astype(int)
            top_box = _find_top_face(hsv[:, :, 2], contour)
            if top_box is None:
                top_xy = contact_xy = _rect_contact_xy(box)
            else:
                top_center = _centroid(top_box)
                blob_center = _centroid(cv2.convexHull(contour))
                top_xy = tuple(float(v) for v in top_center)
                contact_xy = tuple(float(v) for v in 2 * blob_center - top_center)
            detections.append(
                {
                    "label": f"{color_name} {index}",
                    "color": color_name,
                    "bgr": COLOR_BGR[color_name],
                    "box": box,
                    "center_xy": (float(cx), float(cy)),
                    "bbox": (int(left), int(top), int(right), int(bottom)),
                    "top_found": top_box is not None,
                    "top_box": top_box,
                    "top_xy": top_xy,
                    "contact_xy": contact_xy,
                }
            )
    return (detections, mask) if include_mask else detections
