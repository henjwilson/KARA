import numpy as np
import cv2

from cube_pose import refine_blocks


COLOR_RANGES = {
    # Wider HSV range tolerates warm/cool lighting and slightly faded yellow.
    "yellow": (((15, 80, 70), (42, 255, 255)),),
    # Red wraps around the end of the hue circle.
    "red": (((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))),
    # The green blocks are teal: hue 65-87, with blue starting at 89.
    "green": (((45, 70, 60), (88, 255, 255)),),
    "blue": (((89, 80, 60), (130, 255, 255)),),
}

COLOR_BGR = {
    "yellow": (0, 255, 255),
    "red": (60, 60, 255),
    "green": (120, 220, 80),
    "blue": (255, 170, 60),
}

BLOCK_COLORS = tuple(COLOR_RANGES)


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
# Frames this far (deg) from the rest don't count towards a block's angle.
YAW_OUTLIER_DEG = 5.0
# A top face looks about square from above; a longer shape has a side in it.
TOP_FACE_MAX_ASPECT = 1.15
TOP_FACE_THRESHOLD_STEP = 6
# Share of a blob's largest inner radius a part must exceed to count as the
# thick middle of a block when looking for touching blocks.
SPLIT_THICKNESS = 0.7


def _rect_contact_xy(box):
    """Midpoint of the rectangle's lowest edge in the image."""
    order = np.argsort(box[:, 1])
    bottom = box[order[-2:]]
    return float(bottom[:, 0].mean()), float(bottom[:, 1].mean())


def _centroid(points):
    m = cv2.moments(points)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def _bright_region(roi, blob, threshold):
    """Outline of the largest part of the blob brighter than ``threshold``."""
    bright = ((roi > threshold) & (blob > 0)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    faces, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(faces, key=cv2.contourArea) if faces else None


def _find_top_face(value, contour):
    """Rotated box and outline of the block's top face, or None if it can't be
    told apart.

    Lit from above, the top face is the brightest part of the blob, so the
    blob's pixels are split by brightness (Otsu) and the largest bright
    region is taken. A side facing the light can pass that split too, which
    stretches the region, so brighter splits are tried until it looks square.
    """
    x, y, w, h = cv2.boundingRect(contour)
    blob = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blob, [contour - (x, y)], -1, 255, -1)
    roi = value[y:y + h, x:x + w]
    pixels = roi[blob > 0].reshape(-1, 1)
    otsu, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blob_area = max(cv2.contourArea(contour), 1.0)
    best = None
    for threshold in np.arange(otsu, float(pixels.max()), TOP_FACE_THRESHOLD_STEP):
        face = _bright_region(roi, blob, threshold)
        if face is None:
            break
        share = cv2.contourArea(face) / blob_area
        if not TOP_FACE_AREA_RANGE[0] <= share <= TOP_FACE_AREA_RANGE[1]:
            continue
        (_, _), (width, height), _ = cv2.minAreaRect(face)
        aspect = max(width, height) / max(min(width, height), 1.0)
        if best is None or aspect < best[0]:
            best = (aspect, face)
        if aspect <= TOP_FACE_MAX_ASPECT:
            break
    if best is None:
        return None
    face = best[1] + (x, y)
    return cv2.boxPoints(cv2.minAreaRect(face)).astype(np.float32), face.reshape(-1, 2)


def _split_touching(frame, contour):
    """Contours of the blocks in a blob; one, unless blocks are touching.

    A lone block's blob has one thick middle. Touching blocks have one each
    with a neck between, so keeping only the blob's thickest parts leaves
    separate pieces, which seed a watershed along the real edge between them.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 2
    blob = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    cv2.drawContours(blob, [contour - (x - pad, y - pad)], -1, 255, -1)
    distance = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    thick = (distance > SPLIT_THICKNESS * distance.max()).astype(np.uint8)
    count, seeds = cv2.connectedComponents(thick)
    if count <= 2:
        return [contour]
    roi = cv2.copyMakeBorder(frame[y:y + h, x:x + w], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    markers = seeds.astype(np.int32) + 1
    markers[(blob > 0) & (thick == 0)] = 0
    cv2.watershed(roi, markers)
    parts = []
    for label in range(2, count + 1):
        part = ((markers == label) & (blob > 0)).astype(np.uint8)
        found, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        parts.append(max(found, key=cv2.contourArea) + (x - pad, y - pad))
    return parts


def detect_colored_blocks(
    frame,
    min_area=400,
    border=None,
    border_margin=0.0,
    min_fill=0.6,
    colors=("yellow",),
    camera=None,
):
    """Return blocks of the given colors with their top-face centers and table points.

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

    for color_name in colors:
        ranges = COLOR_RANGES[color_name]
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
        for contour in (part for blob in contours if cv2.contourArea(blob) >= min_area
                        for part in _split_touching(frame, blob)):
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
            top_face = _find_top_face(hsv[:, :, 2], contour)
            top_box, outline = (None, contour.reshape(-1, 2)) if top_face is None else top_face
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
                    "outline": outline,
                    "contour": contour.reshape(-1, 2),
                    "top_xy": top_xy,
                    "contact_xy": contact_xy,
                }
            )
    if camera is None:
        return detections
    return refine_blocks(detections, frame, hsv[:, :, 2], camera)


def block_yaw(block, to_table):
    """Direction of a block's edges on the table, in robot degrees [0, 90).

    Its four sides look alike, so an angle only matters modulo 90. The outline
    is fitted on the table, not in the image: the camera looks at the table at
    a slant, so a turned square is a skewed shape in the image. ``to_table``
    maps an Nx2 array of image pixels to robot XY. A fitted cube is a true
    square on the table, so one of its edges is enough.
    """
    if "base_box" in block:
        (ax, ay), (bx, by) = to_table(np.asarray(block["base_box"][:2], dtype=np.float64))
        return float(np.degrees(np.arctan2(by - ay, bx - ax))) % 90.0
    table_xy = to_table(np.asarray(block["outline"], dtype=np.float32))
    return cv2.minAreaRect(np.asarray(table_xy, dtype=np.float32))[2] % 90.0


def _wrapped(angles, center):
    """Signed difference from ``center``, for angles that repeat every 90 deg."""
    return (np.asarray(angles, dtype=float) - center + 45.0) % 90.0 - 45.0


def mean_yaw(yaws, weights=None):
    """Average of block angles, which wrap around every 90 degrees."""
    angles = np.radians(np.asarray(yaws, dtype=float) * 4.0)
    weights = np.ones(len(angles)) if weights is None else np.asarray(weights, dtype=float)
    return float(np.degrees(np.arctan2((weights * np.sin(angles)).sum(),
                                       (weights * np.cos(angles)).sum())) / 4.0) % 90.0


def robust_yaw(yaws, weights):
    """Average angle without outliers, its spread, and how many frames agree."""
    yaws, weights = np.asarray(yaws, dtype=float), np.asarray(weights, dtype=float)
    middle = mean_yaw(yaws)
    keep = np.abs(_wrapped(yaws, middle)) <= YAW_OUTLIER_DEG
    if not keep.any() or weights[keep].sum() <= 0:
        return middle, 90.0, 0
    center = mean_yaw(yaws[keep], weights[keep])
    return center, float(np.abs(_wrapped(yaws[keep], center)).max()), int(keep.sum())
