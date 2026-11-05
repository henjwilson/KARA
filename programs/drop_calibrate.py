"""Collect camera/robot calibration pairs across the table and gripper angles.

Start ``parol6-server`` in another terminal, put the yellow block in the
gripper, then run this from the repo's ``.venv``:

    .venv/bin/python programs/drop_calibrate.py --camera 0

Each run makes 30 drops it hasn't logged yet, so running it again adds new
ones, and a run you quit carries on where it stopped. Every spot is dropped at
3 gripper headings in a row, since the arm's error changes with heading. Two in
three spots are in the green area of OpenCV/reach_map.py, dropped at the
heading seek turns the gripper around there and 30 deg either side. The rest
are outside it, spread over whatever headings each can reach, tilted only if it
reaches no other way.

For each point:

1. The arm moves to the point at ``--safe-height`` above the table, with the
   gripper turned to that point's angle, then comes straight down
   to ``--stop-height``. Each orientation is checked for a straight descent to
   the table before the run starts; the plan is printed first.
2. Hold Q / E to lower / raise it until the block just touches the table, and
   J / L, I / K or U / O to turn the gripper about its tip; it stops when you
   let go, and [ / ] change the speed. H homes, G re-references the joints
   against their limit switches, and either comes back to this point. V opens
   and B closes the gripper. Enter releases; S skips the point.
3. The gripper opens, the arm lifts straight up and homes out of view, and
   the camera measures the block (median over ``MEASURE_SAMPLES`` frames),
   reporting its position and angle against what was commanded.
4. The release pose (``robot``) and the camera position (``camera``) are
   appended to ``cartesian_coords.csv``, the measured angle as the camera
   row's ``ry_deg``.
5. The arm returns above the release pose, comes straight down, grips the
   block, and homes.

Keys are read from the camera window, so click it first. Esc quits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OPENCV_DIR = Path(__file__).resolve().parent.parent / "OpenCV"
sys.path.insert(0, str(OPENCV_DIR))

from pinokin import se3_from_rpy  # noqa: E402

import parol6.PAROL6_ROBOT as PAROL6_ROBOT  # noqa: E402
from parol6 import RobotClient  # noqa: E402
from parol6.config import LIMITS  # noqa: E402
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
    HEADING_STEP_DEG,
    _heading,
    preferred_heading,
    HOME_ANGLES,
    heading_window,
    lifts_straight_up,
    plan_heading,
    set_gripper,
    table_z,
)
from camera_station import HeldKeys  # noqa: E402
from cartesian_keyboard import (  # noqa: E402
    JOG_ACCEL,
    JOG_DURATION,
    JOG_KEYS,
    LINEAR_MAX_MM_S,
    _clear_cartesian_stream,
    keyboard,
)
from cube_pose import CUBE_SIDE_MM, CubeCamera  # noqa: E402
from reach_map import border_extension, test_spots, work_border  # noqa: E402
from table_fit import project  # noqa: E402
from util import block_yaw, detect_colored_blocks, mean_yaw  # noqa: E402

WINDOW = "Drop calibrate"
CSV_HEADER = ("timestamp", "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg", "source", "level")

# Slow, gently accelerated moves: hard starts and stops make the steppers skip,
# which shifts the arm a little more on every trip.
MOVE_SPEED = 0.15
MOVE_ACCEL = 0.3
# Near the table, and while carrying or releasing the block, move slowly.
SLOW_SPEED = 0.06
# Speeds in mm/s for Q / E, and deg/s for the rotation keys.
LOWER_SPEEDS = (2.0, 5.0, 10.0, 20.0)
DEFAULT_POINTS = 30
# One point in this many is outside the green area.
OUTSIDE_EVERY = 3
# Headings per spot, around the one seek uses in the green area, and across the
# reachable range outside it.
GREEN_HEADING_OFFSETS = (-30.0, 0.0, 30.0)
OUTSIDE_HEADING_FRACTIONS = (0.15, 0.5, 0.85)
# A logged drop this close in heading (deg) counts as that drop.
SAME_HEADING_DEG = 12.0
# Outside the area a point is tilted only if vertical gives it less than this
# many degrees of headings.
MIN_OUTSIDE_RANGE = 30.0
MAX_OUTSIDE_TILT = 40
# A logged release this close (mm) means a point has been tested.
SAME_POINT_MM = 8.0
# Q stops this far below the table plane, which is only known to a few mm.
FLOOR_BELOW_TABLE_MM = 25.0
# How long the arm keeps moving after a jog ends, which Q has to stop ahead of.
COAST_S = 0.25
LOWER_KEYS = ("q", "e")
ROTATE_KEYS = ("j", "l", "i", "k", "u", "o")
ANGULAR_MAX_DEG_S = math.degrees(LIMITS.cart.jog.velocity.angular)
# Camera samples (top face found) whose median is logged, and how far from
# the release point a detection may be to count as this block.
MEASURE_SAMPLES = 30
SEARCH_RADIUS_MM = 60.0
MEASURE_TIMEOUT_S = 10.0
FIRST_FRAME_ATTEMPTS = 30
HOME_TIMEOUT_S = 120.0
REGRIP_SETTLE_S = 1.5

ENTER_KEYS = (13, 10)
ESC_KEY = 27


class QuitRequested(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--dictionary", default=None,
                        help="ArUco dictionary (default: the calibration's)")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS,
                        help="New points to test this run")
    parser.add_argument("--safe-height", type=float, default=50.0,
                        help="TCP height above the table when travelling (mm)")
    parser.add_argument("--stop-height", type=float, default=10.0,
                        help="TCP height above the table where it waits to be lowered (mm)")
    parser.add_argument("--level", type=int, default=0,
                        help="Cubes under the release; put them there by hand at each point")
    parser.add_argument("--exposure", type=int, default=None,
                        help="Manual camera exposure (default: leave as it is)")
    parser.add_argument("--focus", type=int, default=None,
                        help="Manual camera focus (default: leave as it is)")
    return parser.parse_args()


def tested_drops(level: int) -> list[tuple[float, float, float]]:
    """Release positions and headings already logged at this level."""
    if not COORDS_CSV.exists():
        return []
    with COORDS_CSV.open(newline="", encoding="utf-8") as coords_file:
        return [(float(r["x_mm"]), float(r["y_mm"]),
                 _heading([float(r[k]) for k in ("rx_deg", "ry_deg", "rz_deg")]))
                for r in csv.DictReader(coords_file)
                if r.get("source") == "robot" and r.get("level") not in (None, "")
                and int(r["level"]) == level]


def outside_range(x: float, y: float, table: float,
                  safe_height: float) -> tuple[float, float, int] | None:
    """The headings a point outside the area can reach: low, high and tilt."""
    best = None
    for tilt in range(0, MAX_OUTSIDE_TILT + 1, 5):
        low, width = heading_window(x, y, table, safe_height, tilt)
        if width >= 0 and (best is None or width > best[1] - best[0]):
            best = low, low + width, tilt
        if width >= MIN_OUTSIDE_RANGE:
            break
    return best


def spot_plans(x: float, y: float, table: float, safe_height: float,
               outside: bool) -> list[tuple[Plan, float, float]]:
    """A plan for each heading a spot is dropped at, with the spot's heading range."""
    if outside:
        reach = outside_range(x, y, table, safe_height)
        if reach is None:
            return []
        low, high, tilt = reach
        headings = [low + (high - low) * f for f in OUTSIDE_HEADING_FRACTIONS]
    else:
        _, width = heading_window(x, y, table, safe_height)
        if width < 90.0:
            return []
        middle, tilt = preferred_heading(x, y, table, safe_height), 0
        low, high = middle - 45.0, middle + 45.0
        headings = [middle + offset for offset in GREEN_HEADING_OFFSETS]
    plans = []
    for heading in headings:
        # The range was found on a coarse grid, so check the heading itself.
        snapped = HEADING_STEP_DEG * round(heading / HEADING_STEP_DEG)
        plan = (plan_heading(x, y, table, safe_height, heading, tilt)
                or plan_heading(x, y, table, safe_height, snapped, tilt))
        if plan is not None:
            plans.append((plan, low, high))
    return plans


def test_points(vision: "Vision", count: int, safe_height: float,
                level: int) -> list[tuple[float, float, Plan, float, float, bool]]:
    """The next ``count`` drops with nothing logged at them yet."""
    inside, outside = (iter(spots) for spots in test_spots(vision.calibration))
    tested = tested_drops(level)
    chosen = []
    spots = 0
    while len(chosen) < count:
        spots += 1
        is_outside = spots % OUTSIDE_EVERY == 0
        spot = next(outside if is_outside else inside, None)
        if spot is None:
            break
        x, y = spot
        table = table_z(vision.z_plane, x, y) + level * CUBE_SIDE_MM
        for plan, low, high in spot_plans(x, y, table, safe_height, is_outside):
            heading = plan.ry_range[0]
            if any(math.dist(spot, done[:2]) <= SAME_POINT_MM
                   and abs((done[2] - heading + 180.0) % 360.0 - 180.0) <= SAME_HEADING_DEG
                   for done in tested):
                continue
            if len(chosen) < count:
                chosen.append((x, y, plan, low, high, is_outside))
    return chosen


def apply_camera_settings(exposure: int | None, focus: int | None) -> None:
    if exposure is None and focus is None:
        return
    from camera_controls import open_controls

    controls = open_controls()
    if controls is None:
        return
    if exposure is not None:
        controls.set_exposure(exposure)
    if focus is not None:
        controls.set_focus(focus)
    print(f"Camera now at exposure {controls.exposure}, focus {controls.focus}.")
    controls.close()


def append_rows(rows: list[tuple]) -> None:
    if not COORDS_CSV.exists() or COORDS_CSV.stat().st_size == 0:
        COORDS_CSV.write_text(",".join(CSV_HEADER) + "\n", encoding="utf-8")
    else:
        text = COORDS_CSV.read_text(encoding="utf-8")
        header, _, rest = text.partition("\n")
        if "level" not in header:
            text = f"{header.rstrip()},level\n{rest}"
        if not text.endswith("\n"):
            text += "\n"
        COORDS_CSV.write_text(text, encoding="utf-8")
    with COORDS_CSV.open("a", newline="", encoding="utf-8") as coords_file:
        csv.writer(coords_file).writerows(rows)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Vision:
    def __init__(self, camera_index: int, dictionary: str | None,
                 exposure: int | None = None, focus: int | None = None) -> None:
        data = json.loads(DEFAULT_CALIBRATION.read_text(encoding="utf-8"))
        width, height = (int(v) for v in data["image_size"])
        self.dictionary = dictionary or data["dictionary"]
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise SystemExit(f"Could not open camera {camera_index}.")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        frame = self.first_frame()
        apply_camera_settings(exposure, focus)
        self.calibration = load_calibration(
            DEFAULT_CALIBRATION, (frame.shape[1], frame.shape[0]), self.dictionary
        )
        self.detector = make_aruco_detector(self.dictionary)
        self.homography = self.calibration["pixel_to_robot_xy"]
        self.cube_camera = CubeCamera(self.homography, (frame.shape[1], frame.shape[0]))
        self.z_plane = np.asarray(self.calibration["table_z_plane"], dtype=float)
        self.extension = border_extension(self.calibration)
        self.border = work_border(cv2.convexHull(np.asarray(
            [m["image_xy"] for m in self.calibration["markers"]], dtype=np.float32
        )), self.extension)
        self.shift_warned = False

    def first_frame(self) -> np.ndarray:
        # A camera just released by another program can take a moment to deliver.
        for _ in range(FIRST_FRAME_ATTEMPTS):
            ok, frame = self.camera.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.1)
        raise SystemExit("Camera opened but did not provide a frame.")

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

    def measure(self, near_xy: tuple[float, float],
                level: int) -> tuple[float, float, float] | None:
        """Median robot XY and average angle of the yellow block at ``level``
        nearest ``near_xy``, or None."""
        for _ in range(5):  # drop frames buffered while the arm was moving
            self.camera.grab()
        samples: list[tuple[float, float]] = []
        yaws: list[float] = []
        deadline = time.monotonic() + MEASURE_TIMEOUT_S
        while len(samples) < MEASURE_SAMPLES and time.monotonic() < deadline:
            frame = self.read()
            markers = detect_markers(frame, self.detector)
            self.shift_warned = check_camera_shift(markers, self.calibration,
                                                   self.shift_warned)
            self.border = work_border(aruco_border(markers, self.border), self.extension)
            view = frame.copy()
            marker_overlay(view, markers)
            candidates = []
            for block in detect_colored_blocks(frame, border=self.border,
                                               border_margin=30,
                                               camera=self.cube_camera):
                if not block["top_found"] or block["level"] != level:
                    continue
                xy = map_xy(block["contact_xy"], self.homography)
                distance = math.dist(xy, near_xy)
                if distance <= SEARCH_RADIUS_MM:
                    candidates.append((distance, xy, block))
            if candidates:
                _, xy, block = min(candidates, key=lambda c: c[0])
                samples.append(xy)
                yaws.append(block_yaw(block, lambda pixels: project(self.homography, pixels)))
                cv2.polylines(view, [np.rint(block["top_box"]).astype(np.int32)],
                              True, (0, 255, 255), 2)
            self.show([f"Measuring block: {len(samples)}/{MEASURE_SAMPLES}"], view)
        if len(samples) < MEASURE_SAMPLES:
            return None
        x, y = np.median(np.asarray(samples), axis=0)
        return float(x), float(y), mean_yaw(yaws)

    def close(self) -> None:
        self.camera.release()
        cv2.destroyAllWindows()


def move_to(rbt: RobotClient, pose: list[float], speed: float) -> None:
    rbt.move_l(pose, frame="WRF", speed=speed, accel=MOVE_ACCEL, wait=True, timeout=60.0)


def go_home(rbt: RobotClient) -> None:
    """Back to the home pose by a slow joint move, rather than the controller's
    home command at full speed."""
    rbt.move_j(HOME_ANGLES, speed=MOVE_SPEED, accel=MOVE_ACCEL, wait=True, timeout=60.0)


def settled_pose(rbt: RobotClient) -> list[float] | None:
    """The pose once a jog has run out."""
    _clear_cartesian_stream(rbt)
    return rbt.pose(frame="WRF")


def lower_by_hand(rbt: RobotClient, vision: Vision, held: HeldKeys, label: str,
                  floor_z: float, lift_z: float,
                  rehome: Callable[[bool], None]) -> list[float] | None:
    """Let the user drive the arm down; returns the release pose, or None to skip."""
    speed_index = LOWER_SPEEDS.index(5.0)
    held.drop()
    moving = None
    warning = ""
    while True:
        pose = rbt.pose(frame="WRF")
        height = pose[2] - (floor_z + FLOOR_BELOW_TABLE_MM) if pose else float("nan")
        speed = LOWER_SPEEDS[speed_index]
        key = vision.show([
            label,
            f"TCP {height:6.1f} mm above where it meets the table   {speed:g} mm/s or deg/s",
            warning or "Hold Q E  J L  I K  U O   [ ] speed   H home  G limit switches   "
                       "V open  B close   Enter release  S skip  Esc quit",
        ])
        if 0 < key < 128 and chr(key) in (*LOWER_KEYS, *ROTATE_KEYS):
            held.window_press(chr(key))
        elif key == ord("["):
            speed_index = max(0, speed_index - 1)
        elif key == ord("]"):
            speed_index = min(len(LOWER_SPEEDS) - 1, speed_index + 1)
        elif key in (ord("v"), ord("b")):
            held.drop()
            moving = None
            settled_pose(rbt)
            try:
                set_gripper(rbt, close=key == ord("b"))
            except GripperError as exc:
                print(f"Gripper failed: {exc}")
            continue
        elif key in (ord("h"), ord("g")):
            held.drop()
            moving = None
            at = settled_pose(rbt)
            # Going home from low would drag the block, and the limit-switch
            # sequence ignores everything in its way.
            if at is None or not lifts_straight_up(rbt.angles(), at, lift_z):
                warning = "Can't lift straight up from here to home: turn back first"
                print(warning)
                continue
            warning = ""
            move_to(rbt, [*at[:2], max(at[2], lift_z), *at[3:]], SLOW_SPEED)
            rehome(key == ord("g"))
            held.drop()
            continue
        elif key in ENTER_KEYS or key == ord("s"):
            held.drop()
            release = settled_pose(rbt)
            if key == ord("s"):
                return None
            if release is None:
                print("Pose unavailable; try again.")
                continue
            # After letting go the arm has to rise straight up, which it can't
            # from every pose near the table outside the green area.
            if lifts_straight_up(rbt.angles(), release, lift_z):
                return release
            warning = "Can't lift straight up from here, still holding: turn back or S skip"
            print(warning)
            continue
        active = held.active_jog_key()
        if active == "q" and pose is not None and pose[2] - speed * COAST_S <= floor_z:
            active = None
        if active != moving:
            moving = active
            if active:
                _clear_cartesian_stream(rbt)
        if active in LOWER_KEYS:
            direction = -1.0 if active == "q" else 1.0
            rbt.jog_l("WRF", "Z", direction * speed / LINEAR_MAX_MM_S,
                      duration=JOG_DURATION, accel=JOG_ACCEL)
        elif active:
            axis, direction, frame = JOG_KEYS[active]
            rbt.jog_l(frame, axis, direction * speed / ANGULAR_MAX_DEG_S,
                      duration=JOG_DURATION, accel=JOG_ACCEL)


def released_angle(plan: Plan, release: list[float]) -> float:
    """Block angle commanded at release: the plan's, plus any turning by hand."""
    pose = np.zeros((4, 4))
    se3_from_rpy(0.0, 0.0, 0.0, *np.radians(release[3:]), pose)
    turned = pose[:3, :3] @ plan.rotation.T
    return (plan.ry_range[0] + math.degrees(math.atan2(turned[1, 0], turned[0, 0]))) % 90.0


def measure_until_found(vision: Vision, near_xy: tuple[float, float],
                        level: int) -> tuple[float, float] | None:
    while True:
        found = vision.measure(near_xy, level)
        if found is not None:
            return found
        print("Block not seen clearly. Enter: try again, S: don't log this point.")
        while True:
            key = vision.show(["Block not seen: Enter retry, S skip logging"])
            if key in ENTER_KEYS:
                break
            if key == ord("s"):
                return None


def run_point(rbt: RobotClient, vision: Vision, held: HeldKeys, number: int, total: int,
              x: float, y: float, plan: Plan, level: int = 0,
              stop_drop: float = 40.0) -> tuple[float, float] | None:
    """Run one point; returns the camera's position and angle error, if it saw it."""
    angles = plan.angles
    print(f"\nPoint {number}/{total}: X={x:.1f} Y={y:.1f}  {plan.describe()}")
    rbt.move_j(angles, speed=MOVE_SPEED, accel=MOVE_ACCEL, wait=True, timeout=60.0)
    safe_pose = rbt.pose(frame="WRF")
    def go_to_stop() -> None:
        # Arriving high keeps the gripper clear of anything on the table on its way.
        rbt.move_j(angles, speed=MOVE_SPEED, accel=MOVE_ACCEL, wait=True, timeout=60.0)
        move_to(rbt, [*safe_pose[:2], safe_pose[2] - stop_drop, *safe_pose[3:]], SLOW_SPEED)

    def rehome(limit_switches: bool) -> None:
        print("Re-referencing the joints..." if limit_switches else "Homing...")
        if limit_switches:
            rbt.home(wait=True, calibrate=True, timeout=HOME_TIMEOUT_S)
        else:
            go_home(rbt)
        go_to_stop()

    go_to_stop()

    below = f" onto {level} cube(s) placed under it" if level else ""
    floor_z = table_z(vision.z_plane, x, y) + level * CUBE_SIDE_MM - FLOOR_BELOW_TABLE_MM
    release = lower_by_hand(rbt, vision, held, f"Point {number}/{total}: lower the block{below}",
                            floor_z, safe_pose[2], rehome)
    if release is None:
        print("Skipped; keeping the block.")
        pose = rbt.pose(frame="WRF")
        if pose is not None:
            move_to(rbt, [*pose[:2], safe_pose[2], *pose[3:]], SLOW_SPEED)
        move_to(rbt, safe_pose, SLOW_SPEED)
        go_home(rbt)
        return

    set_gripper(rbt, close=False)
    move_to(rbt, [*release[:2], safe_pose[2], *release[3:]], SLOW_SPEED)
    go_home(rbt)

    found = measure_until_found(vision, (release[0], release[1]), level)
    result = None
    if found is not None:
        cx, cy, yaw = found
        stamp = timestamp()
        append_rows([
            (stamp, *release, "robot", level),
            (stamp, cx, cy, table_z(vision.z_plane, cx, cy), "", yaw, "", "camera", level),
        ])
        error = math.dist((cx, cy), release[:2])
        commanded = released_angle(plan, release)
        angle_error = (yaw - commanded + 45.0) % 90.0 - 45.0
        result = error, angle_error
        print(f"Released at X={release[0]:.1f} Y={release[1]:.1f}; camera saw "
              f"X={cx:.1f} Y={cy:.1f} (off by {error:.1f} mm), angle {yaw:.1f} deg "
              f"against {commanded:.1f} commanded ({angle_error:+.1f}). Logged.")

    # Line up above the block first: any rotation must be done before going down.
    above_release = [*release[:2], safe_pose[2], *release[3:]]
    rbt.move_j(angles, speed=MOVE_SPEED, accel=MOVE_ACCEL, wait=True, timeout=60.0)
    move_to(rbt, above_release, SLOW_SPEED)
    move_to(rbt, release, SLOW_SPEED)
    # Let the arm stop swaying so the jaws close square on the block.
    time.sleep(REGRIP_SETTLE_S)
    set_gripper(rbt, close=True)
    move_to(rbt, above_release, SLOW_SPEED)
    go_home(rbt)
    return result


def print_summary(results: list[tuple[float, float]]) -> None:
    if not results:
        return
    errors = np.asarray(results)
    angles = errors[:, 1]
    print(f"\n{len(results)} points measured. Position: {errors[:, 0].mean():.1f} mm on "
          f"average, {errors[:, 0].max():.1f} worst. Angle: {angles.mean():+.1f} deg offset, "
          f"{np.abs(angles - angles.mean()).max():.1f} worst around it.")


def main() -> None:
    args = parse_args()
    PAROL6_ROBOT.apply_tool(TOOL, TOOL_VARIANT)
    vision = Vision(args.camera, args.dictionary, args.exposure, args.focus)
    print("Planning orientations...")
    points = test_points(vision, args.points, args.safe_height, args.level)
    if not points:
        raise SystemExit("No new points to test. Build OpenCV/reach_map.py if it is missing.")
    if len(points) < args.points:
        print(f"Only {len(points)} of {args.points} new points are left.")
    print(f"{len(points)} points. Rows go to {COORDS_CSV}.")
    for number, (x, y, plan, low, high, outside) in enumerate(points, start=1):
        where = "outside" if outside else "green"
        print(f"  {number:2d}: {where:7} X={x:6.1f} Y={y:7.1f} heading range {low:5.0f} to "
              f"{high:5.0f}  {plan.describe()}")

    held = HeldKeys()
    listener = keyboard.Listener(on_press=held.on_press, on_release=held.on_release)
    listener.start()
    results = []
    rbt = RobotClient(host=HOST, port=PORT)
    try:
        tool_index = rbt.select_tool(TOOL, TOOL_VARIANT)
        if tool_index >= 0:
            rbt.wait_command(tool_index)
        set_gripper(rbt, close=True)
        print("Homing...")
        rbt.home(wait=True)
        for number, (x, y, plan, *_) in enumerate(points, start=1):
            try:
                result = run_point(rbt, vision, held, number, len(points), x, y, plan,
                                   args.level, args.safe_height - args.stop_height)
            except (MotionError, TimeoutError, GripperError) as exc:
                print(f"Failed at point {number} ({exc}). Stopping; run again once "
                      "it's clear and it carries on from here.")
                rbt.stop()
                return
            if result is not None:
                results.append(result)
        print("\nAll points done.")
    except GripperError as exc:
        print(f"Gripper failed before starting: {exc}")
    except QuitRequested:
        print("Quit. The arm was left where it is.")
        rbt.stop()
    finally:
        print_summary(results)
        listener.stop()
        vision.close()
        rbt.close()


if __name__ == "__main__":
    main()
