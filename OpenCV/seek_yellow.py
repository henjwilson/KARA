"""Watch the table for yellow blocks; hover the robot above each one, then home.

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script
    from the repo's `.venv` (it needs both parol6 and OpenCV). Add
    `--calibrate` to calibrate the camera first (same as main.py's):

        .venv/bin/python OpenCV/seek_yellow.py --camera 0 --calibrate

Each cycle: wait for a confirmed yellow block inside the ArUco border, move
straight to its XY at TARGET_HEIGHT_MM above the calibrated table surface
there (the table isn't level in robot coordinates), wait DWELL_S, and home. Blocks
already visited are skipped until they move. Press Q in the camera window to
stop.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6 import RobotClient
from parol6.utils.errors import MotionError
from parol6.utils.ik import solve_ik

from main import (
    DEFAULT_CALIBRATION,
    DICTIONARIES,
    BlockTracker,
    add_calibration_args,
    aruco_border,
    check_camera_shift,
    detect_markers,
    load_calibration,
    make_aruco_detector,
    map_xy,
    marker_overlay,
    run_calibration,
)
from util import detect_colored_blocks

HOST, PORT = "127.0.0.1", 5001
TOOL, TOOL_VARIANT = "PNEUMATIC", "vertical"
HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]

TARGET_HEIGHT_MM = 30.0  # TCP height above the table surface
DWELL_S = 2.0
MOVE_SPEED = 0.3
# A block within this distance of one already visited is the same block.
VISITED_RADIUS_MM = 25.0
# A visited spot empty for this many frames is forgotten, so a new block
# placed there gets picked up.
FORGET_AFTER_FRAMES = 30

# Gripper attitude used when the calibration markers were touched; the
# headings searched below rotate it about the vertical axis only.
_TOUCH_ROTATION = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
HEADING_OFFSETS_DEG = sorted(range(-180, 180, 10), key=abs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dictionary", choices=DICTIONARIES, default="4X4_50")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--min-area", type=float, default=400)
    parser.add_argument("--border-margin", type=float, default=30)
    parser.add_argument("--min-fill", type=float, default=0.6)
    parser.add_argument("--confirm-frames", type=int, default=10,
                        help="Frames a block must be seen before the robot goes to it")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect and solve IK, but do not move the robot")
    add_calibration_args(parser)
    return parser.parse_args()


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def solve_hover(x: float, y: float, z: float) -> list[float] | None:
    """Joint angles (deg) putting the TCP at (x, y, z) mm, or None if unreachable.

    The gripper's touch attitude points it toward the robot base; the heading
    nearest that which IK can reach (seeded from home) is used.
    """
    seed = np.radians(HOME_ANGLES)
    # _TOUCH_ROTATION's tool-Z column is -Y, i.e. heading -90 deg.
    base_heading = math.degrees(math.atan2(-y, -x)) + 90.0
    target = np.eye(4)
    target[:3, 3] = (x / 1000.0, y / 1000.0, z / 1000.0)
    for offset in HEADING_OFFSETS_DEG:
        target[:3, :3] = _rot_z(base_heading + offset) @ _TOUCH_ROTATION
        result = solve_ik(PAROL6_ROBOT.robot, target, seed, quiet_logging=True)
        if result.success:
            return np.degrees(result.q).tolist()
    return None


def table_z(z_plane: np.ndarray, x: float, y: float) -> float:
    return float(z_plane[0] * x + z_plane[1] * y + z_plane[2])


def visit(rbt: RobotClient | None, x: float, y: float, z: float,
          angles: list[float]) -> None:
    """Move straight to the block's target pose, wait, and home."""
    print(f"Going to yellow at X={x:.1f} Y={y:.1f} Z={z:.1f} mm")
    if rbt is None:
        time.sleep(DWELL_S)
        return
    try:
        rbt.move_j(angles, speed=MOVE_SPEED, wait=True)
        time.sleep(DWELL_S)
    except (MotionError, TimeoutError) as exc:
        print(f"Move failed ({exc}); returning home.")
    finally:
        rbt.home(wait=True)
        print("Home. Looking for another yellow block...")


def main() -> None:
    args = parse_args()
    PAROL6_ROBOT.apply_tool(TOOL, TOOL_VARIANT)
    detector = make_aruco_detector(args.dictionary)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = camera.read()
    if not ok or frame is None:
        camera.release()
        raise SystemExit("Camera opened but did not provide a frame.")

    if args.calibrate and not run_calibration(
        camera, frame, detector, args.calibration, args.dictionary, args.marker_size
    ):
        camera.release()
        cv2.destroyAllWindows()
        return
    calibration = load_calibration(
        args.calibration, (frame.shape[1], frame.shape[0]), args.dictionary
    )
    rbt = None
    if not args.dry_run:
        rbt = RobotClient(host=HOST, port=PORT)
        tool_index = rbt.select_tool(TOOL, TOOL_VARIANT)
        if tool_index >= 0:
            rbt.wait_command(tool_index)
        print("Homing...")
        rbt.home(wait=True)
    homography = calibration["pixel_to_robot_xy"]
    z_plane = np.asarray(calibration["table_z_plane"], dtype=float)
    border = cv2.convexHull(np.asarray(
        [m["image_xy"] for m in calibration["markers"]], dtype=np.float32
    ))
    tracker = BlockTracker(args.confirm_frames)
    visited: dict[tuple[float, float], int] = {}  # spot -> frames since last seen
    unreachable: list[tuple[float, float]] = []
    robot_job: threading.Thread | None = None
    shift_warned = False
    print("Looking for yellow blocks. Press Q in the camera window to quit.")

    try:
        while True:
            detections = detect_markers(frame, detector)
            shift_warned = check_camera_shift(detections, calibration, shift_warned)
            border = aruco_border(detections, border)
            busy = robot_job is not None and robot_job.is_alive()
            # The arm passes through the view while moving; restart
            # confirmation once it is home so its motion can't trigger a visit.
            if busy:
                tracker.tracks.clear()
            raw = [] if busy else detect_colored_blocks(
                frame, min_area=args.min_area, border=border,
                border_margin=args.border_margin, min_fill=args.min_fill,
            )
            blocks = tracker.update(raw)
            if not busy:
                seen = [map_xy(b["contact_xy"], homography) for b in raw]
                for spot in list(visited):
                    if any(math.dist(spot, s) <= VISITED_RADIUS_MM for s in seen):
                        visited[spot] = 0
                    elif visited[spot] >= FORGET_AFTER_FRAMES:
                        del visited[spot]
                    else:
                        visited[spot] += 1

            view = frame.copy()
            marker_overlay(view, detections)
            cv2.polylines(view, [np.rint(border).astype(np.int32)], True, (255, 160, 0), 2)
            target = None
            for block in blocks:
                x, y = map_xy(block["contact_xy"], homography)
                done = any(math.dist((x, y), p) <= VISITED_RADIUS_MM
                           for p in [*visited, *unreachable])
                if block["top_found"]:
                    outline, color, dot = block["top_box"], block["bgr"], (255, 255, 255)
                else:
                    outline, color, dot = block["box"], (0, 140, 255), (0, 140, 255)
                if done:
                    color = (128, 128, 128)
                cv2.polylines(view, [np.rint(outline).astype(np.int32)], True, color, 2)
                tx, ty = block["top_xy"]
                cv2.circle(view, (round(tx), round(ty)), 5, dot, -1)
                left, top, _, _ = block["bbox"]
                mode = "top" if block["top_found"] else "no top face"
                cv2.putText(view, f"({mode}) X{x:.0f} Y{y:.0f}", (left, max(20, top - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if not done and target is None:
                    target = (x, y)
            status = "ROBOT MOVING" if busy else "SEARCHING"
            cv2.putText(view, f"{status} | visited {len(visited)}", (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Seek yellow", view)

            if target is not None:
                x, y = target
                z = table_z(z_plane, x, y) + TARGET_HEIGHT_MM
                angles = solve_hover(x, y, z)
                if angles is None:
                    print(f"Yellow at X={x:.1f} Y={y:.1f} is out of reach; skipping.")
                    unreachable.append(target)
                else:
                    visited[target] = 0
                    robot_job = threading.Thread(
                        target=visit, args=(rbt, x, y, z, angles), daemon=True
                    )
                    robot_job.start()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame lost.")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if rbt is not None:
            if robot_job is not None and robot_job.is_alive():
                rbt.stop()
            rbt.close()


if __name__ == "__main__":
    main()
