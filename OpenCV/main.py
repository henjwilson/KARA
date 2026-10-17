"""Map ArUco markers and colored blocks from camera pixels into robot WRF XY."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from table_fit import PAIRING_MARGIN, fit_similarity, fit_table, is_mirrored, project
from util import detect_colored_blocks


DEFAULT_CALIBRATION = Path(__file__).with_name("table_calibration.json")
COORDS_CSV = Path(__file__).resolve().parent.parent / "programs" / "cartesian_coords.csv"
# A camera row and a robot row this close (mm) are the same block.
REFINE_PAIR_MM = 40.0
REFINE_MIN_PAIRS = 3
DICTIONARIES = ("4X4_50", "4X4_100", "5X5_100", "6X6_250", "7X7_1000")
MAP_SIZE = (900, 700)
MAP_MARGIN = 55
CALIBRATION_FRAMES = 15
# Warn when a calibration marker has moved this far in the image.
CAMERA_SHIFT_WARN_PX = 15.0

# Robot WRF millimeters of each marker's center, touched with the
# PNEUMATIC/vertical gripper's TCP held just above the marker. Seen from the
# camera: ID 8 top-left, 11 top-right, 5 bottom-right, 2 bottom-left (so
# image-up is robot -Y and image-right is robot -X).
MARKER_ROBOT_COORDS_MM = {
    2: (299.5026823564, 199.6258653900, -2.1627053417),
    5: (182.9104654491, 202.1685145635, -11.7663270022),
    8: (321.6584876099, -254.5970431422, 7.1079269738),
    11: (215.2683896891, -261.5579851145, -8.8318130756),
}
# How far above the table surface the TCP was when touching the markers; the
# fitted table plane is this much below the touched points.
TOUCH_GAP_MM = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map table ArUco markers and colored blocks to robot coordinates."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument(
        "--dictionary", choices=DICTIONARIES, default="4X4_50",
        help="ArUco dictionary used by the printed markers (default: 4X4_50)",
    )
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION,
        help=f"Calibration JSON (default: {DEFAULT_CALIBRATION})",
    )
    parser.add_argument(
        "--min-area", type=float, default=400,
        help="Minimum yellow contour area in pixels (default: 400)",
    )
    parser.add_argument(
        "--confirm-frames", type=int, default=3,
        help="Consecutive frames a block must be seen before it is reported "
             "(default: 3)",
    )
    parser.add_argument(
        "--border-margin", type=float, default=30,
        help="Pixels outside the ArUco border a block center may sit and still "
             "be detected (default: 30)",
    )
    parser.add_argument(
        "--min-fill", type=float, default=0.6,
        help="Minimum fraction of its fitted rectangle a yellow region must fill; "
             "lower accepts more irregular shapes (default: 0.6)",
    )
    add_calibration_args(parser)
    return parser.parse_args()


def add_calibration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Calibrate using the preset WRF coordinates for ArUco IDs 2, 5, 8, and 11",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="Correct the calibration using camera/robot point pairs saved by "
             "cartesian_keyboard (Y then C on the same blocks), then exit",
    )
    parser.add_argument(
        "--marker-size", type=float, default=None,
        help="Printed ArUco marker side length in mm (black border edge to edge). "
             "Optional; if omitted it is measured from the touched points",
    )


def make_aruco_detector(dictionary_name: str):
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise SystemExit(
            "cv2.aruco is missing. Install the OpenCV contrib package with: "
            "python -m pip install -r OpenCV/requirements.txt"
        )
    dictionary = aruco.getPredefinedDictionary(
        getattr(aruco, f"DICT_{dictionary_name}")
    )
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary)
    parameters = (
        aruco.DetectorParameters()
        if hasattr(aruco, "DetectorParameters")
        else aruco.DetectorParameters_create()
    )
    return dictionary, parameters


def detect_markers(frame: np.ndarray, detector) -> list[dict]:
    if hasattr(detector, "detectMarkers"):
        corners, ids, _rejected = detector.detectMarkers(frame)
    else:
        dictionary, parameters = detector
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            frame, dictionary, parameters=parameters
        )
    if ids is None:
        return []
    detections = []
    for marker_id, corner_set in zip(ids.flatten().tolist(), corners):
        points = np.asarray(corner_set, dtype=np.float32).reshape(4, 2)
        center = points.mean(axis=0)
        detections.append({"id": int(marker_id), "corners": points, "center": center})
    return detections


def marker_overlay(frame: np.ndarray, detections: list[dict]) -> None:
    for marker in detections:
        pts = np.rint(marker["corners"]).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cx, cy = np.rint(marker["center"]).astype(int)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            frame, f"ID {marker['id']}", (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 255, 30), 2,
        )


def calibrate(
    frame_size: tuple[int, int],
    corners_by_id: dict[int, np.ndarray],
    path: Path,
    dictionary_name: str,
    marker_size_mm: float | None = None,
) -> None:
    missing = sorted(set(MARKER_ROBOT_COORDS_MM) - set(corners_by_id))
    if missing:
        raise ValueError(
            f"Could not see calibrated marker ID(s) {missing}. Put all four "
            "calibrated markers in view, then press C."
        )
    ids = sorted(MARKER_ROBOT_COORDS_MM)
    corners = np.asarray([corners_by_id[i] for i in ids], dtype=np.float64)
    robot = np.asarray([MARKER_ROBOT_COORDS_MM[i] for i in ids], dtype=np.float64)
    centers_px = corners.mean(axis=1)
    if cv2.contourArea(cv2.convexHull(centers_px.astype(np.float32))) < 100:
        raise ValueError("Calibration markers are too close together in the image.")
    if cv2.contourArea(cv2.convexHull(robot[:, :2].astype(np.float32))) < 100:
        raise ValueError("Robot marker coordinates are too close/collinear.")

    print("Fitting table geometry (checking all marker pairings)...")
    given, best = fit_table(corners, robot[:, :2], marker_size_mm)
    fit = given
    if best.pairing != given.pairing and best.cost * PAIRING_MARGIN < given.cost:
        fit = best
        robot = robot[list(best.pairing)]
        print("WARNING: the preset marker coordinates look mixed up. They fit the "
              "camera geometry much better paired like this:")
        for marker_id, source in zip(ids, best.pairing):
            if ids[source] != marker_id:
                print(f"  ID {marker_id} <- coordinates listed for ID {ids[source]}")
        print("Using the corrected pairing. Fix MARKER_ROBOT_COORDS_MM in main.py "
              "to match.")
    elif best.pairing != given.pairing:
        # The touched points are too symmetric for the camera to tell these
        # pairings apart, so the configured one is kept.
        print("NOTE: the camera can't confirm the marker pairing; this one fits "
              "almost as well:")
        for marker_id, source in zip(ids, best.pairing):
            if ids[source] != marker_id:
                print(f"  ID {marker_id} <- coordinates listed for ID {ids[source]}")
        print("If the robot goes to the wrong spot, swap MARKER_ROBOT_COORDS_MM "
              "in main.py to that pairing.")

    homography = fit.homography
    denominators = centers_px @ homography[2, :2] + homography[2, 2]
    if float(denominators.min()) * float(denominators.max()) <= 0:
        raise ValueError("Calibration folds the camera view; check the markers lie flat.")
    if is_mirrored(homography, centers_px.mean(axis=0)):
        raise ValueError("Calibration came out mirrored; check marker coordinates.")

    heights = robot[:, 2] - TOUCH_GAP_MM
    z_design = np.column_stack((robot[:, 0], robot[:, 1], np.ones(len(robot))))
    z_plane, *_ = np.linalg.lstsq(z_design, heights, rcond=None)
    z_rms_mm = float(np.sqrt(np.mean((heights - z_design @ z_plane) ** 2)))
    touch_rms_mm = float(np.sqrt(np.mean(fit.touch_residuals_mm ** 2)))

    marker_records = []
    for index, marker_id in enumerate(ids):
        values = robot[index]
        print(f"ArUco ID {marker_id}: robot X={values[0]:.1f} Y={values[1]:.1f} "
              f"Z={values[2]:.1f} mm | camera puts it "
              f"{fit.touch_residuals_mm[index]:.1f} mm from there")
        marker_records.append({
            "id": marker_id,
            "image_xy": centers_px[index].tolist(),
            "image_corners": corners[index].tolist(),
            "robot_xyz_mm": values.tolist(),
            "fitted_xy_mm": fit.centers_mm[index].tolist(),
        })

    calibration = {
        "version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_size": list(frame_size),
        "dictionary": dictionary_name,
        "pixel_to_robot_xy": homography.tolist(),
        "table_z_mm": float(np.mean(heights)),
        "table_z_plane": z_plane.tolist(),
        "table_z_rms_mm": z_rms_mm,
        "markers": marker_records,
        "marker_size_mm": fit.marker_size_mm,
        "touch_rms_mm": touch_rms_mm,
        "corner_rms_mm": fit.corner_rms_mm,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    size_note = "given" if marker_size_mm else "measured — compare with a ruler"
    print(f"Saved calibration to {path}")
    print(f"Marker size: {fit.marker_size_mm:.1f} mm ({size_note})")
    print(f"Touched points vs camera geometry: {touch_rms_mm:.1f} mm RMS")
    print(f"Marker squareness error: {fit.corner_rms_mm:.2f} mm RMS")
    print(f"Table-height plane residual: {z_rms_mm:.2f} mm RMS")
    if touch_rms_mm > 10:
        print("WARNING: the touched robot coordinates disagree with the camera by "
              "more than 10 mm. Re-touch the marker centers (check the tool/TCP "
              "is set correctly) for better accuracy.")
    if fit.corner_rms_mm > 2:
        print("WARNING: the markers don't look like flat, equal squares. Make sure "
              "they lie flat and are all printed at the same size.")
    print("Keep the camera fixed and use this same camera resolution for tracking.")


def run_calibration(
    camera: cv2.VideoCapture,
    frame: np.ndarray,
    detector,
    path: Path,
    dictionary_name: str,
    marker_size_mm: float | None,
) -> bool:
    """Show the calibration view until the user calibrates (True) or quits."""
    frame_size = (int(frame.shape[1]), int(frame.shape[0]))
    print(f"Calibration view {frame_size[0]}x{frame_size[1]}: show ArUco IDs "
          f"{sorted(MARKER_ROBOT_COORDS_MM)}, then press C. Press Q to quit.")
    samples: dict[int, list[np.ndarray]] | None = None
    while True:
        detections = detect_markers(frame, detector)
        visible = {m["id"]: m["corners"] for m in detections
                   if m["id"] in MARKER_ROBOT_COORDS_MM}
        if samples is not None and len(visible) == len(MARKER_ROBOT_COORDS_MM):
            for marker_id, corners in visible.items():
                samples[marker_id].append(corners)
            if len(samples[marker_id]) >= CALIBRATION_FRAMES:
                averaged = {i: np.mean(c, axis=0) for i, c in samples.items()}
                samples = None
                try:
                    calibrate(frame_size, averaged, path, dictionary_name, marker_size_mm)
                except ValueError as exc:
                    print(f"Calibration failed: {exc}")
                else:
                    return True
        view = frame.copy()
        marker_overlay(view, detections)
        status = ("Hold still: averaging frames..." if samples is not None
                  else f"Detected {len(visible)}/{len(MARKER_ROBOT_COORDS_MM)} "
                       "calibration markers | press C to calibrate")
        cv2.putText(view, status, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (30, 255, 30), 2)
        cv2.imshow("ArUco table calibration", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("c") and samples is None:
            if len(visible) < len(MARKER_ROBOT_COORDS_MM):
                missing = sorted(set(MARKER_ROBOT_COORDS_MM) - set(visible))
                print(f"Can't calibrate yet: marker ID(s) {missing} not visible.")
            else:
                samples = {i: [] for i in MARKER_ROBOT_COORDS_MM}
        ok, frame = camera.read()
        if not ok or frame is None:
            print("Camera frame lost.")
            return False


def load_calibration(
    path: Path, frame_size: tuple[int, int], dictionary_name: str
) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"No calibration found at {path}. First run: "
            f"python OpenCV/main.py --calibrate"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    saved = sorted(tuple(np.round(m["robot_xyz_mm"], 2)) for m in data.get("markers", []))
    expected = sorted(tuple(np.round(v, 2)) for v in MARKER_ROBOT_COORDS_MM.values())
    if data.get("version") != 3 or saved != expected:
        raise SystemExit(
            "This calibration is from an older calibration method or marker "
            "coordinates. Recalibrate with: python OpenCV/main.py --calibrate --camera 0"
        )
    saved_size = tuple(int(value) for value in data["image_size"])
    if saved_size != frame_size:
        raise SystemExit(
            f"Camera frame is {frame_size[0]}x{frame_size[1]}, but calibration was "
            f"made at {saved_size[0]}x{saved_size[1]}. Recalibrate at this resolution "
            "or use the original --width and --height."
        )
    if data.get("dictionary") != dictionary_name:
        raise SystemExit(
            f"Calibration uses DICT_{data.get('dictionary')}; rerun with "
            f"--dictionary {data.get('dictionary')} or recalibrate."
        )
    data["pixel_to_robot_xy"] = np.asarray(
        data["pixel_to_robot_xy"], dtype=np.float64
    )
    return data


def refine_calibration(path: Path, coords_csv: Path = COORDS_CSV) -> None:
    """Correct the pixel -> robot map with block positions measured both ways.

    Pairs each ``camera`` row saved since the map last changed with the
    nearest ``robot`` row (the TCP touched to that block), then fits a
    rotation + scale + shift taking camera XY onto robot XY. Few pairs can't
    support a freer correction without overfitting.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    since = datetime.fromisoformat(data.get("updated_utc", data["created_utc"]))
    with coords_csv.open(newline="", encoding="utf-8") as coords_file:
        rows = [r for r in csv.DictReader(coords_file) if r.get("timestamp")]
    robot = [r for r in rows if r.get("source") == "robot"]
    camera = [r for r in rows if r.get("source") == "camera"
              and datetime.fromisoformat(r["timestamp"]) >= since]
    skipped = sum(1 for r in rows if r.get("source") == "camera") - len(camera)
    if skipped:
        print(f"Ignoring {skipped} camera row(s) taken before the calibration "
              "last changed (they used an older map).")

    def xy(row: dict) -> np.ndarray:
        return np.array([float(row["x_mm"]), float(row["y_mm"])])

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    unused = list(robot)
    for row in camera:
        if not unused:
            break
        nearest = min(unused, key=lambda r: float(np.linalg.norm(xy(r) - xy(row))))
        if np.linalg.norm(xy(nearest) - xy(row)) <= REFINE_PAIR_MM:
            unused.remove(nearest)
            pairs.append((xy(row), xy(nearest)))
    if len(pairs) < REFINE_MIN_PAIRS:
        raise ValueError(
            f"Found {len(pairs)} camera/robot pair(s); need at least "
            f"{REFINE_MIN_PAIRS}. For each block press Y, then touch its top "
            "center with the TCP and press C, in cartesian_keyboard."
        )
    cam_xy = np.array([p[0] for p in pairs])
    robot_xy = np.array([p[1] for p in pairs])
    before = np.linalg.norm(robot_xy - cam_xy, axis=1)

    correction = fit_similarity(cam_xy, robot_xy)
    after = np.linalg.norm(project(correction, cam_xy) - robot_xy, axis=1)
    held_out = []
    for i in range(len(pairs)):
        keep = [j for j in range(len(pairs)) if j != i]
        fit = fit_similarity(cam_xy[keep], robot_xy[keep])
        held_out.append(float(np.linalg.norm(project(fit, cam_xy[i:i + 1])[0] - robot_xy[i])))

    print(f"Using {len(pairs)} camera/robot pairs:")
    for (c, r), e0, e1 in zip(pairs, before, after):
        print(f"  camera ({c[0]:6.1f}, {c[1]:7.1f}) robot ({r[0]:6.1f}, {r[1]:7.1f}) "
              f"| error {e0:5.1f} -> {e1:4.1f} mm")
    angle = np.degrees(np.arctan2(correction[1, 0], correction[0, 0]))
    scale = float(np.hypot(correction[0, 0], correction[1, 0]))
    print(f"Correction: rotate {angle:+.2f} deg, scale {scale:.4f}, "
          f"shift ({correction[0, 2]:+.1f}, {correction[1, 2]:+.1f}) mm")
    print(f"Error: {before.mean():.1f} mm before -> {after.mean():.1f} mm after "
          f"(on points left out of the fit: {np.mean(held_out):.1f} mm)")
    if np.mean(held_out) >= before.mean():
        raise ValueError("The correction doesn't help on held-out points; "
                         "calibration left unchanged. Record more pairs.")

    homography = correction @ np.asarray(data["pixel_to_robot_xy"])
    homography /= homography[2, 2]
    data.setdefault("pixel_to_robot_xy_markers", data["pixel_to_robot_xy"])
    data["pixel_to_robot_xy"] = homography.tolist()
    data["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data.setdefault("refinements", []).append({
        "updated_utc": data["updated_utc"],
        "pairs": [{"camera_xy_mm": c.tolist(), "robot_xy_mm": r.tolist()} for c, r in pairs],
        "correction": correction.tolist(),
        "mean_error_before_mm": float(before.mean()),
        "mean_error_after_mm": float(after.mean()),
        "held_out_error_mm": float(np.mean(held_out)),
    })
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved refined calibration to {path}. Camera rows saved from now on "
          "use the refined map; older ones are ignored by the next --refine.")


def check_camera_shift(detections: list[dict], calibration: dict, warned: bool) -> bool:
    """Warn once if a calibration marker moved in the image since calibrating.

    Returns whether the warning has been shown.
    """
    saved = {m["id"]: np.asarray(m["image_xy"]) for m in calibration["markers"]}
    shifts = [float(np.linalg.norm(m["center"] - saved[m["id"]]))
              for m in detections if m["id"] in saved]
    if warned or not shifts or max(shifts) <= CAMERA_SHIFT_WARN_PX:
        return warned
    print(f"WARNING: a calibration marker moved {max(shifts):.0f} px since "
          "calibrating (camera or marker bumped?). Positions may be off; "
          "recalibrate with --calibrate.")
    return True


def aruco_border(detections: list[dict], fallback: np.ndarray) -> np.ndarray:
    """Convex hull around the four calibration markers' outer corners.

    Uses the live markers when all four are visible, otherwise ``fallback``
    (the last good border) so a briefly occluded marker doesn't drop blocks.
    """
    corners = [
        marker["corners"] for marker in detections
        if marker["id"] in MARKER_ROBOT_COORDS_MM
    ]
    if len(corners) < len(MARKER_ROBOT_COORDS_MM):
        return fallback
    return cv2.convexHull(np.concatenate(corners).astype(np.float32))


class BlockTracker:
    """Suppress flickering noise by only reporting blocks seen in several
    consecutive frames near the same image position."""

    def __init__(self, confirm_frames: int, max_jump_px: float = 25.0):
        self.confirm_frames = confirm_frames
        self.max_jump_px = max_jump_px
        self.tracks: list[dict] = []

    def update(self, blocks: list[dict]) -> list[dict]:
        unmatched = list(self.tracks)
        tracks = []
        for block in blocks:
            center = np.asarray(block["center_xy"])
            hits = 1
            if unmatched:
                distances = [np.linalg.norm(t["center"] - center) for t in unmatched]
                nearest = int(np.argmin(distances))
                if distances[nearest] <= self.max_jump_px:
                    hits = unmatched.pop(nearest)["hits"] + 1
            tracks.append({"center": center, "hits": hits, "block": block})
        self.tracks = tracks
        return [t["block"] for t in tracks if t["hits"] >= self.confirm_frames]


def map_xy(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    transformed = cv2.perspectiveTransform(
        np.asarray([[[point[0], point[1]]]], dtype=np.float32), homography
    )[0, 0]
    return float(transformed[0]), float(transformed[1])


def create_virtual_space(calibration: dict, blocks: list[dict]) -> np.ndarray:
    """Draw an XY view whose pixel scale is uniform and derived from mm."""
    width, height = MAP_SIZE
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    calibration_xy = np.asarray(
        [m["robot_xyz_mm"][:2] for m in calibration["markers"]], dtype=float
    )
    x_span = max(float(np.ptp(calibration_xy[:, 0])), 100.0)
    y_span = max(float(np.ptp(calibration_xy[:, 1])), 100.0)
    x_pad, y_pad = max(50.0, x_span * 0.1), max(50.0, y_span * 0.1)
    x_min = float(calibration_xy[:, 0].min()) - x_pad
    x_max = float(calibration_xy[:, 0].max()) + x_pad
    y_min = float(calibration_xy[:, 1].min()) - y_pad
    y_max = float(calibration_xy[:, 1].max()) + y_pad
    scale = min(
        (width - 2 * MAP_MARGIN) / max(x_max - x_min, 1.0),
        (height - 2 * MAP_MARGIN) / max(y_max - y_min, 1.0),
    )
    offset_x = (width - (x_max - x_min) * scale) / 2.0
    offset_y = (height - (y_max - y_min) * scale) / 2.0

    def screen_xy(x: float, y: float) -> tuple[int, int]:
        return (
            round(offset_x + (x - x_min) * scale),
            round(height - offset_y - (y - y_min) * scale),
        )

    tick = 50
    for x in np.arange(np.ceil(x_min / tick) * tick, x_max, tick):
        sx, _ = screen_xy(float(x), y_min)
        cv2.line(canvas, (sx, 0), (sx, height), (52, 52, 52), 1)
        cv2.putText(canvas, f"{x:.0f}", (sx + 2, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    for y in np.arange(np.ceil(y_min / tick) * tick, y_max, tick):
        _, sy = screen_xy(x_min, float(y))
        cv2.line(canvas, (0, sy), (width, sy), (52, 52, 52), 1)
        cv2.putText(canvas, f"{y:.0f}", (4, sy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    cv2.putText(canvas, "Virtual table | robot WRF (mm)", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
    cv2.putText(canvas, f"Scale: {1.0 / scale:.2f} mm/pixel", (18, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    for marker in calibration["markers"]:
        x, y = marker["robot_xyz_mm"][:2]
        point = screen_xy(x, y)
        cv2.drawMarker(canvas, point, (130, 130, 130), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(canvas, f"A{marker['id']}", (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    for block in blocks:
        point = screen_xy(block["x"], block["y"])
        cv2.circle(canvas, point, max(5, round(8 / (1.0 / scale))), block["bgr"], -1)
        cv2.circle(canvas, point, 7, (245, 245, 245), 1)
        label = f"{block['label']} ({block['x']:.0f}, {block['y']:.0f})"
        cv2.putText(canvas, label, (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    return canvas


def print_blocks(blocks: list[dict], z_plane: np.ndarray) -> None:
    if not blocks:
        print("No colored blocks detected.")
        return
    print("Colored block coordinates (robot WRF, mm):")
    for block in blocks:
        z = float(z_plane[0] * block["x"] + z_plane[1] * block["y"] + z_plane[2])
        print(
            f"  {block['label']}: X={block['x']:.1f}, Y={block['y']:.1f}, "
            f"Z={z:.1f}"
        )


def main() -> None:
    args = parse_args()
    if args.refine:
        try:
            refine_calibration(args.calibration)
        except (OSError, KeyError, ValueError) as exc:
            raise SystemExit(f"Refine failed: {exc}") from exc
        return
    detector = make_aruco_detector(args.dictionary)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try --camera 1 (or another index)."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, first_frame = camera.read()
    if not ok or first_frame is None:
        camera.release()
        raise SystemExit("Camera opened but did not provide a frame.")
    frame_size = (int(first_frame.shape[1]), int(first_frame.shape[0]))

    try:
        if args.calibrate:
            if run_calibration(camera, first_frame, detector, args.calibration,
                               args.dictionary, args.marker_size):
                print("Calibration complete. Start again without --calibrate to track blocks.")
            return

        calibration = load_calibration(args.calibration, frame_size, args.dictionary)
        homography = calibration["pixel_to_robot_xy"]
        z_plane = np.asarray(calibration["table_z_plane"], dtype=float)
        print(
            f"Tracking with {len(calibration['markers'])} calibration markers; "
            f"average table Z={calibration['table_z_mm']:.1f} mm. "
            "Press P to print coordinates, Q to quit."
        )
        border = cv2.convexHull(np.asarray(
            [m["image_xy"] for m in calibration["markers"]], dtype=np.float32
        ))
        tracker = BlockTracker(args.confirm_frames)
        frame = first_frame
        last_report = 0.0
        last_signature = None
        shift_warned = False
        while True:
            detections = detect_markers(frame, detector)
            shift_warned = check_camera_shift(detections, calibration, shift_warned)
            border = aruco_border(detections, border)
            view = frame.copy()
            marker_overlay(view, detections)
            cv2.polylines(view, [np.rint(border).astype(np.int32)], True, (255, 160, 0), 2)
            image_blocks = detect_colored_blocks(
                frame,
                min_area=args.min_area,
                border=border,
                border_margin=args.border_margin,
                min_fill=args.min_fill,
            )
            image_blocks = tracker.update(image_blocks)
            blocks = []
            for image_block in image_blocks:
                x, y = map_xy(image_block["contact_xy"], homography)
                block = {**image_block, "x": x, "y": y}
                blocks.append(block)
                left, top, _right, _bottom = image_block["bbox"]
                if image_block["top_found"]:
                    top_box = np.rint(image_block["top_box"]).astype(np.int32)
                    cv2.polylines(view, [top_box], True, image_block["bgr"], 2)
                    tx, ty = image_block["top_xy"]
                    cv2.circle(view, (round(tx), round(ty)), 5, (255, 255, 255), -1)
                else:
                    box = np.rint(image_block["box"]).astype(np.int32)
                    cv2.polylines(view, [box], True, (0, 140, 255), 2)
                    px, py = image_block["contact_xy"]
                    cv2.circle(view, (round(px), round(py)), 5, (0, 140, 255), -1)
                mode = "top" if image_block["top_found"] else "no top face"
                text = f"{image_block['label']} ({mode}) X{x:.1f} Y{y:.1f} mm"
                cv2.putText(view, text, (left, max(20, top - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(
                view,
                f"Yellow detections: {len(blocks)}",
                (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
            )
            cv2.imshow("Camera: ArUco + colored blocks", view)
            cv2.imshow("Calibrated robot table space", create_virtual_space(calibration, blocks))

            signature = tuple(
                (b["label"], round(b["x"], 1), round(b["y"], 1)) for b in blocks
            )
            now = time.monotonic()
            if blocks and signature != last_signature and now - last_report >= 0.5:
                print_blocks(blocks, z_plane)
                last_signature = signature
                last_report = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                print_blocks(blocks, z_plane)
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame lost.")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
