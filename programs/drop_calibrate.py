"""Collect camera/robot calibration pairs by setting a yellow block down on a grid.

Start ``parol6-server`` in another terminal, put the yellow block in the
gripper, then run this from the repo's ``.venv``:

    .venv/bin/python programs/drop_calibrate.py --camera 0

For each grid point inside the calibration markers (visited in random order):

1. The arm moves to the point at ``--safe-height`` above the table with Rx
   at 90, where a range of at least 45 deg of Ry (heading) reaches the
   table there; otherwise Rx is lowered as little as needed to get that
   range. It uses the middle of the range. Each orientation is checked for
   a straight descent to the table before the run starts; the plan is
   printed first.
2. You lower it with Q / E (down / up) until the block just touches the
   table. [ / ] change the step size. Enter releases; S skips the point.
3. The gripper opens, the arm lifts straight up and homes out of view, and
   the camera measures the block (median over ``MEASURE_SAMPLES`` frames).
4. The release pose (``robot``) and the camera position (``camera``) are
   appended to ``cartesian_coords.csv``; run ``OpenCV/main.py --refine``
   afterwards to fit them.
5. The arm returns to the release pose, grips the block, and homes.

Keys are read from the camera window, so click it first. Esc quits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OPENCV_DIR = Path(__file__).resolve().parent.parent / "OpenCV"
sys.path.insert(0, str(OPENCV_DIR))

import parol6.PAROL6_ROBOT as PAROL6_ROBOT  # noqa: E402
from parol6 import RobotClient  # noqa: E402
from parol6.utils.errors import MotionError  # noqa: E402

from main import (  # noqa: E402
    COORDS_CSV,
    DEFAULT_CALIBRATION,
    aruco_border,
    check_camera_shift,
    detect_markers,
    load_calibration,
    make_aruco_detector,
    map_xy,
    marker_overlay,
)
from placement import (  # noqa: E402
    HOST,
    PORT,
    TOOL,
    TOOL_VARIANT,
    GripperError,
    Plan,
    plan_point,
    set_gripper,
    table_z,
)
from util import detect_colored_blocks  # noqa: E402

WINDOW = "Drop calibrate"
CSV_HEADER = ("timestamp", "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg", "source")

MOVE_SPEED = 0.3
# Near the table, and while carrying or releasing the block, move slowly.
SLOW_SPEED = 0.1
STEP_SIZES_MM = (0.5, 1.0, 2.0, 5.0, 10.0)
# Grid points must be at least this far inside the marker centers.
INSET_MM = 15.0
# Camera samples (top face found) whose median is logged, and how far from
# the release point a detection may be to count as this block.
MEASURE_SAMPLES = 30
SEARCH_RADIUS_MM = 60.0
MEASURE_TIMEOUT_S = 10.0

ENTER_KEYS = (13, 10)
ESC_KEY = 27


class QuitRequested(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--dictionary", default=None,
                        help="ArUco dictionary (default: the calibration's)")
    parser.add_argument("--spacing", type=float, default=40.0,
                        help="Grid spacing in mm")
    parser.add_argument("--safe-height", type=float, default=50.0,
                        help="TCP height above the table when travelling (mm)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shuffle seed for the visiting order")
    parser.add_argument("--start", type=int, default=1,
                        help="Point number to start from (to resume a run)")
    return parser.parse_args()


def grid_points(calibration: dict, spacing: float) -> list[tuple[float, float]]:
    """Grid over the area inside the calibration markers, in robot XY mm."""
    corners = np.asarray([m["robot_xyz_mm"][:2] for m in calibration["markers"]],
                         dtype=np.float32)
    hull = cv2.convexHull(corners)
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    xs = np.arange(x0 + INSET_MM, x1 - INSET_MM + 1e-6, spacing)
    ys = np.arange(y0 + INSET_MM, y1 - INSET_MM + 1e-6, spacing)
    # Center the grid in the span instead of leaving all the slack at one end.
    xs += (x1 - INSET_MM - xs[-1]) / 2
    ys += (y1 - INSET_MM - ys[-1]) / 2
    return [
        (float(x), float(y)) for x in xs for y in ys
        if cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= INSET_MM
    ]


def append_rows(rows: list[tuple]) -> None:
    if not COORDS_CSV.exists() or COORDS_CSV.stat().st_size == 0:
        COORDS_CSV.write_text(",".join(CSV_HEADER) + "\n", encoding="utf-8")
    elif not COORDS_CSV.read_text(encoding="utf-8").endswith("\n"):
        with COORDS_CSV.open("a", encoding="utf-8") as coords_file:
            coords_file.write("\n")
    with COORDS_CSV.open("a", newline="", encoding="utf-8") as coords_file:
        csv.writer(coords_file).writerows(rows)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Vision:
    def __init__(self, camera_index: int, dictionary: str | None) -> None:
        data = json.loads(DEFAULT_CALIBRATION.read_text(encoding="utf-8"))
        width, height = (int(v) for v in data["image_size"])
        self.dictionary = dictionary or data["dictionary"]
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise SystemExit(f"Could not open camera {camera_index}.")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = self.read()
        self.calibration = load_calibration(
            DEFAULT_CALIBRATION, (frame.shape[1], frame.shape[0]), self.dictionary
        )
        self.detector = make_aruco_detector(self.dictionary)
        self.homography = self.calibration["pixel_to_robot_xy"]
        self.z_plane = np.asarray(self.calibration["table_z_plane"], dtype=float)
        self.border = cv2.convexHull(np.asarray(
            [m["image_xy"] for m in self.calibration["markers"]], dtype=np.float32
        ))
        self.shift_warned = False

    def read(self) -> np.ndarray:
        ok, frame = self.camera.read()
        if not ok or frame is None:
            raise SystemExit("Camera frame lost.")
        return frame

    def show(self, lines: list[str], frame: np.ndarray | None = None) -> int:
        """Show the live view with instructions; returns the key pressed."""
        if frame is None:
            frame = self.read()
        view = frame.copy()
        for row, text in enumerate(lines):
            cv2.putText(view, text, (18, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ESC_KEY:
            raise QuitRequested
        return key

    def measure(self, near_xy: tuple[float, float]) -> tuple[float, float] | None:
        """Median robot XY of the yellow block nearest ``near_xy``, or None."""
        for _ in range(5):  # drop frames buffered while the arm was moving
            self.camera.grab()
        samples: list[tuple[float, float]] = []
        deadline = time.monotonic() + MEASURE_TIMEOUT_S
        while len(samples) < MEASURE_SAMPLES and time.monotonic() < deadline:
            frame = self.read()
            markers = detect_markers(frame, self.detector)
            self.shift_warned = check_camera_shift(markers, self.calibration,
                                                   self.shift_warned)
            self.border = aruco_border(markers, self.border)
            view = frame.copy()
            marker_overlay(view, markers)
            candidates = []
            for block in detect_colored_blocks(frame, border=self.border,
                                               border_margin=30):
                if not block["top_found"]:
                    continue
                xy = map_xy(block["contact_xy"], self.homography)
                distance = math.dist(xy, near_xy)
                if distance <= SEARCH_RADIUS_MM:
                    candidates.append((distance, xy, block))
            if candidates:
                _, xy, block = min(candidates, key=lambda c: c[0])
                samples.append(xy)
                cv2.polylines(view, [np.rint(block["top_box"]).astype(np.int32)],
                              True, (0, 255, 255), 2)
            self.show([f"Measuring block: {len(samples)}/{MEASURE_SAMPLES}"], view)
        if len(samples) < MEASURE_SAMPLES:
            return None
        x, y = np.median(np.asarray(samples), axis=0)
        return float(x), float(y)

    def close(self) -> None:
        self.camera.release()
        cv2.destroyAllWindows()


def move_to(rbt: RobotClient, pose: list[float], speed: float) -> None:
    rbt.move_l(pose, frame="WRF", speed=speed, wait=True, timeout=30.0)


def lower_by_hand(rbt: RobotClient, vision: Vision, label: str) -> list[float] | None:
    """Let the user step the arm down; returns the release pose, or None to skip."""
    step_index = STEP_SIZES_MM.index(2.0)
    while True:
        pose = rbt.pose(frame="WRF")
        height = pose[2] - table_z(vision.z_plane, pose[0], pose[1]) if pose else float("nan")
        step = STEP_SIZES_MM[step_index]
        key = vision.show([
            label,
            f"TCP {height:6.1f} mm above table plane   step {step} mm",
            "Q down  E up  [ ] step  Enter release  S skip  Esc quit",
        ])
        if key in (ord("q"), ord("e")):
            dz = -step if key == ord("q") else step
            try:
                rbt.move_l([0, 0, dz, 0, 0, 0], frame="WRF", rel=True,
                           speed=SLOW_SPEED, wait=True)
            except (MotionError, TimeoutError) as exc:
                print(f"Step failed: {exc}")
        elif key == ord("["):
            step_index = max(0, step_index - 1)
        elif key == ord("]"):
            step_index = min(len(STEP_SIZES_MM) - 1, step_index + 1)
        elif key in ENTER_KEYS:
            if pose is None:
                print("Pose unavailable; try again.")
                continue
            return pose
        elif key == ord("s"):
            return None


def measure_until_found(vision: Vision, near_xy: tuple[float, float]) -> tuple[float, float] | None:
    while True:
        found = vision.measure(near_xy)
        if found is not None:
            return found
        print("Block not seen clearly. Enter: try again, S: don't log this point.")
        while True:
            key = vision.show(["Block not seen: Enter retry, S skip logging"])
            if key in ENTER_KEYS:
                break
            if key == ord("s"):
                return None


def run_point(rbt: RobotClient, vision: Vision, number: int, total: int,
              x: float, y: float, plan: Plan) -> None:
    angles = plan.angles
    print(f"\nPoint {number}/{total}: X={x:.1f} Y={y:.1f}  {plan.describe()}")
    rbt.move_j(angles, speed=MOVE_SPEED, wait=True, timeout=30.0)
    safe_pose = rbt.pose(frame="WRF")

    release = lower_by_hand(rbt, vision, f"Point {number}/{total}: lower the block")
    if release is None:
        print("Skipped; keeping the block.")
        move_to(rbt, safe_pose, SLOW_SPEED)
        rbt.home(wait=True)
        return

    set_gripper(rbt, close=False)
    move_to(rbt, [*release[:2], safe_pose[2], *release[3:]], SLOW_SPEED)
    rbt.home(wait=True)

    found = measure_until_found(vision, (release[0], release[1]))
    if found is not None:
        cx, cy = found
        stamp = timestamp()
        append_rows([
            (stamp, *release, "robot"),
            (stamp, cx, cy, table_z(vision.z_plane, cx, cy), "", "", "", "camera"),
        ])
        error = math.dist((cx, cy), release[:2])
        print(f"Released at X={release[0]:.1f} Y={release[1]:.1f}; camera saw "
              f"X={cx:.1f} Y={cy:.1f} (off by {error:.1f} mm). Logged.")

    rbt.move_j(angles, speed=MOVE_SPEED, wait=True, timeout=30.0)
    move_to(rbt, release, SLOW_SPEED)
    set_gripper(rbt, close=True)
    move_to(rbt, [*release[:2], safe_pose[2], *release[3:]], SLOW_SPEED)
    rbt.home(wait=True)


def main() -> None:
    args = parse_args()
    PAROL6_ROBOT.apply_tool(TOOL, TOOL_VARIANT)
    vision = Vision(args.camera, args.dictionary)
    points = grid_points(vision.calibration, args.spacing)
    random.Random(args.seed).shuffle(points)
    if not points:
        raise SystemExit("No grid points fit inside the markers; lower --spacing.")
    print(f"{len(points)} grid points, {args.spacing:.0f} mm apart. Rows go to {COORDS_CSV}.")
    print("Planning orientations...")
    plans = []
    for number, (x, y) in enumerate(points, start=1):
        plan = plan_point(x, y, table_z(vision.z_plane, x, y), args.safe_height)
        plans.append(plan)
        status = plan.describe() if plan else "UNREACHABLE, will be skipped"
        print(f"  {number:2d}: X={x:6.1f} Y={y:7.1f}  {status}")

    rbt = RobotClient(host=HOST, port=PORT)
    try:
        tool_index = rbt.select_tool(TOOL, TOOL_VARIANT)
        if tool_index >= 0:
            rbt.wait_command(tool_index)
        set_gripper(rbt, close=True)
        print("Homing...")
        rbt.home(wait=True)
        for number, ((x, y), plan) in enumerate(zip(points, plans), start=1):
            if number < args.start or plan is None:
                continue
            try:
                run_point(rbt, vision, number, len(points), x, y, plan)
            except (MotionError, TimeoutError, GripperError) as exc:
                print(f"Failed at point {number} ({exc}). Stopping; resume "
                      f"with --start {number} --seed {args.seed} once it's clear.")
                rbt.stop()
                return
        print("\nAll points done. Fit them with: .venv/bin/python OpenCV/main.py --refine")
    except GripperError as exc:
        print(f"Gripper failed before starting: {exc}")
    except QuitRequested:
        print("Quit. The arm was left where it is.")
        rbt.stop()
    finally:
        vision.close()
        rbt.close()


if __name__ == "__main__":
    main()
