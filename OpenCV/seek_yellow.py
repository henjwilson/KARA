"""Watch the table for yellow blocks; pick each one up and drop it at home.

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script
    from the repo's `.venv` (it needs both parol6 and OpenCV). It needs
    drop-calibrate pairs taken with the current camera calibration, so run
    `programs/drop_calibrate.py` first:

        .venv/bin/python OpenCV/seek_yellow.py --camera 0

Each cycle: wait for a confirmed yellow block inside the ArUco border, turn
its camera position into robot XY with the drop-calibrate correction, and
plan a gripper orientation the same way drop_calibrate.py does. The arm
moves SAFE_HEIGHT_MM above the block, descends straight down to the TCP
height the drop-calibrate releases had there, grips, lifts, homes, and opens
the gripper to drop the block. Press Q in the camera window to stop.
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

from main import (
    COORDS_CSV,
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
from placement import (
    HOST,
    PORT,
    TOOL,
    TOOL_VARIANT,
    GripperError,
    Plan,
    load_drop_correction,
    plan_point,
    set_gripper,
)
from util import detect_colored_blocks

# Matches drop_calibrate.py's default, so picks use the orientations its
# placements were checked with.
SAFE_HEIGHT_MM = 50.0
MOVE_SPEED = 0.3
# Near the table, and while carrying the block, move slowly.
SLOW_SPEED = 0.1
# A block within this distance of one already visited is the same block.
VISITED_RADIUS_MM = 25.0
# A visited spot empty for this many frames is forgotten, so a new block
# placed there gets picked up.
FORGET_AFTER_FRAMES = 30


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
    parser.add_argument("--drop-pairs", type=int, default=23,
                        help="Use only the most recent N drop-calibrate pairs "
                             "(one full drop_calibrate run is 23)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect and plan, but do not move the robot")
    add_calibration_args(parser)
    return parser.parse_args()


def pick_and_drop(rbt: RobotClient | None, x: float, y: float, z: float,
                  plan: Plan) -> None:
    """Pick the block at (x, y) with the TCP at height z, then drop it at home."""
    print(f"Picking yellow at X={x:.1f} Y={y:.1f} Z={z:.1f} mm  {plan.describe()}")
    if rbt is None:
        time.sleep(1.0)
        return
    above = None
    try:
        rbt.move_j(plan.angles, speed=MOVE_SPEED, wait=True, timeout=30.0)
        above = rbt.pose(frame="WRF")
        if above is None:
            raise TimeoutError("pose unavailable above the block")
        rbt.move_l([x, y, z, *above[3:]], speed=SLOW_SPEED, wait=True, timeout=30.0)
        set_gripper(rbt, close=True)
        rbt.move_l(above, speed=SLOW_SPEED, wait=True, timeout=30.0)
        rbt.home(wait=True)
        set_gripper(rbt, close=False)
        print("Dropped at home. Looking for another yellow block...")
    except (MotionError, TimeoutError, GripperError) as exc:
        print(f"Pick failed ({exc}); returning home.")
        try:
            if above is not None:
                rbt.move_l(above, speed=SLOW_SPEED, wait=True, timeout=30.0)
            rbt.home(wait=True)
            set_gripper(rbt, close=False)
        except (MotionError, TimeoutError, GripperError) as home_exc:
            print(f"Could not get home ({home_exc}); stop and check the arm.")


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
    try:
        correction = load_drop_correction(args.calibration, COORDS_CSV, args.drop_pairs)
    except SystemExit:
        camera.release()
        raise
    print(correction.summary())
    rbt = None
    if not args.dry_run:
        rbt = RobotClient(host=HOST, port=PORT)
        tool_index = rbt.select_tool(TOOL, TOOL_VARIANT)
        if tool_index >= 0:
            rbt.wait_command(tool_index)
        print("Homing...")
        rbt.home(wait=True)
        set_gripper(rbt, close=False)
    homography = calibration["pixel_to_robot_xy"]

    def to_robot(pixel_xy: tuple[float, float]) -> tuple[float, float]:
        return correction.to_robot(map_xy(pixel_xy, homography))

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
                seen = [to_robot(b["contact_xy"]) for b in raw]
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
                steady = block["steady_contact_xy"]
                x, y = to_robot(steady or block["contact_xy"])
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
                # A block whose top face is rarely found is shown but not
                # picked: its position is only a rough estimate.
                mode = "top" if steady else "no top face, waiting"
                cv2.putText(view, f"({mode}) X{x:.0f} Y{y:.0f}", (left, max(20, top - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if steady and not done and target is None:
                    target = (x, y)
            status = "ROBOT MOVING" if busy else "SEARCHING"
            cv2.putText(view, f"{status} | visited {len(visited)}", (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Seek yellow", view)

            if target is not None:
                x, y = target
                z = correction.pick_z(x, y)
                plan = plan_point(x, y, z, SAFE_HEIGHT_MM)
                if plan is None:
                    print(f"Yellow at X={x:.1f} Y={y:.1f} is out of reach; skipping.")
                    unreachable.append(target)
                else:
                    visited[target] = 0
                    robot_job = threading.Thread(
                        target=pick_and_drop, args=(rbt, x, y, z, plan), daemon=True
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
