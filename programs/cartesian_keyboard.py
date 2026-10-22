"""Keyboard teleoperation in Cartesian space for the PAROL6 robot.

Start ``parol6-server`` in another terminal, then run this file. Press H to
home before jogging, or G to re-run the calibrated homing (every joint seeks
its limit switch again). Hold one movement key at a time. Press P to print the
current pose or C to save it to ``cartesian_coords.csv`` next to this script.
Press Y to find yellow blocks with the camera (top-face center, mapped to
robot XY with OpenCV/table_calibration.json) and save those to the CSV too;
the ``source`` column tells robot rows from camera rows. Press V to open
the gripper and B to close it.

With Rx free (off by default; set RX_FREE or toggle with R), WRF translation
keys treat Rx as a don't-care: X/Y/Z move while Ry and Rz stay fixed, and Rx
tilts to whatever keeps the target reachable.

macOS may require Accessibility permission for the terminal app running this
script so pynput can read keyboard events.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from pynput import keyboard
except ImportError as exc:
    raise SystemExit(
        "This program needs pynput. Install it with: "
        "uv pip install --python .venv/bin/python pynput"
    ) from exc

import numpy as np
from pinokin import se3_from_rpy

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6 import RobotClient
from parol6.config import LIMITS
from parol6.utils.errors import MotionError
from parol6.utils.ik import solve_ik

HOST, PORT = "127.0.0.1", 5001
CAMERA_INDEX = 0
OPENCV_DIR = Path(__file__).resolve().parent.parent / "OpenCV"
# A yellow block must be seen in this many of the sampled frames to be saved.
YELLOW_FRAMES = 10
YELLOW_MIN_HITS = 6
# Detections this close together (mm) across frames are the same block.
YELLOW_MATCH_MM = 25.0
CSV_HEADER = ("timestamp", "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg", "source")
TOOL, TOOL_VARIANT = "PNEUMATIC", "vertical"

LOOP_HZ = 20
JOG_DURATION = (1.0 / LOOP_HZ) + 0.05
JOG_SPEED = 0.15
JOG_ACCEL = 0.30
TRANSLATION_FRAME = "WRF"

RX_FREE = False
RX_SEARCH_STEP_DEG = 0.5
RX_SEARCH_MAX_DEG = 10.0
# Near wrist singularities IK can "succeed" on a different branch, flipping a
# joint by 90-180°; such solutions are treated as unreachable.
MAX_JOINT_STEP_RAD = math.radians(10.0)
# Caps how far the streamed target may run ahead of the arm, so it stops
# promptly on release even if joint velocity limits slowed it down.
MAX_LEAD_MM = 10.0
LINEAR_MAX_MM_S = LIMITS.cart.jog.velocity.linear * 1000.0
TRANSLATION_INDEX = {"X": 0, "Y": 1, "Z": 2}
RX_OFFSETS = [0.0] + [
    sign * step * RX_SEARCH_STEP_DEG
    for step in range(1, int(RX_SEARCH_MAX_DEG / RX_SEARCH_STEP_DEG) + 1)
    for sign in (1, -1)
]

# Each entry is (axis, direction, frame).
JOG_KEYS = {
    "d": ("X", 1, "WRF"),
    "a": ("X", -1, "WRF"),
    "w": ("Y", 1, "WRF"),
    "s": ("Y", -1, "WRF"),
    "e": ("Z", 1, "WRF"),
    "q": ("Z", -1, "WRF"),
    "j": ("RX", 1, "TRF"),
    "l": ("RX", -1, "TRF"),
    "i": ("RY", 1, "TRF"),
    "k": ("RY", -1, "TRF"),
    "o": ("RZ", 1, "TRF"),
    "u": ("RZ", -1, "TRF"),
}

pressed_keys: set[str] = set()
pending_actions: set[str] = set()
state_lock = threading.Lock()
quit_requested = False
_target_se3 = np.zeros((4, 4), dtype=np.float64)


def _key_name(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if isinstance(key, keyboard.KeyCode):
        return key.char.lower() if key.char else None
    return str(key).replace("Key.", "").lower()


def on_press(key: keyboard.Key | keyboard.KeyCode):
    global quit_requested
    name = _key_name(key)
    if name is None:
        return None
    if name == "esc":
        with state_lock:
            quit_requested = True
        return False
    with state_lock:
        if name not in pressed_keys:
            pressed_keys.add(name)
            if name == "space":
                pressed_keys.difference_update(JOG_KEYS)
            if name in {"h", "g", "p", "c", "y", "v", "b", "[", "]", "-", "=", "f", "r"}:
                pending_actions.add(name)
    return None


def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
    name = _key_name(key)
    if name is not None:
        with state_lock:
            pressed_keys.discard(name)


def _read_state() -> tuple[set[str], set[str], bool]:
    with state_lock:
        keys = set(pressed_keys)
        actions = set(pending_actions)
        pending_actions.clear()
        return keys, actions, quit_requested


def _read_pose(rbt: RobotClient) -> list[float] | None:
    pose = rbt.pose(frame="WRF")
    if pose is None:
        print("Pose unavailable.")
        return None
    print(
        f"Position WRF (mm):  X={pose[0]:8.2f}   Y={pose[1]:8.2f}   Z={pose[2]:8.2f}\n"
        f"Rotation WRF (°):  Rx={pose[3]:8.2f}  Ry={pose[4]:8.2f}  Rz={pose[5]:8.2f}"
    )
    return pose


def _append_rows(rows: list[tuple]) -> Path | None:
    """Append rows to cartesian_coords.csv, adding the header (or the source
    column to an older header) as needed. Returns the path, or None on error."""
    path = Path(__file__).with_name("cartesian_coords.csv")
    try:
        if not path.exists() or path.stat().st_size == 0:
            path.write_text(",".join(CSV_HEADER) + "\n", encoding="utf-8")
        else:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            if lines[0].strip() == ",".join(CSV_HEADER[:-1]):
                lines[0] = ",".join(CSV_HEADER) + "\n"
                path.write_text("".join(lines), encoding="utf-8")
            elif not lines[-1].endswith("\n"):
                with path.open("a", encoding="utf-8") as coords_file:
                    coords_file.write("\n")
        with path.open("a", newline="", encoding="utf-8") as coords_file:
            csv.writer(coords_file).writerows(rows)
    except OSError as exc:
        print(f"Could not save coordinates: {exc}")
        return None
    return path


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _save_pose(rbt: RobotClient) -> None:
    pose = rbt.pose(frame="WRF")
    if pose is None:
        print("Pose unavailable; nothing saved.")
        return
    path = _append_rows([(_timestamp(), *pose, "robot")])
    if path is not None:
        print(f"Coordinates saved to {path}")


class YellowCamera:
    """Camera + table calibration for locating yellow blocks in robot XY.

    Opened on first use and kept open, since opening a camera takes a while.
    """

    def __init__(self) -> None:
        self._camera = None
        self._ready = False

    def _open(self) -> bool:
        if str(OPENCV_DIR) not in sys.path:
            sys.path.insert(0, str(OPENCV_DIR))
        import cv2
        import main as vision

        self._cv2, self._vision = cv2, vision
        try:
            data = json.loads(vision.DEFAULT_CALIBRATION.read_text(encoding="utf-8"))
            width, height = (int(v) for v in data["image_size"])
            dictionary = data["dictionary"]
        except (OSError, KeyError, ValueError) as exc:
            print(f"No usable camera calibration ({exc}). Run: "
                  ".venv/bin/python OpenCV/main.py --calibrate --camera 0")
            return False
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            print(f"Could not open camera {CAMERA_INDEX}; set CAMERA_INDEX.")
            return False
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = camera.read()
        if not ok or frame is None:
            camera.release()
            print("Camera opened but did not provide a frame.")
            return False
        try:
            calibration = vision.load_calibration(
                vision.DEFAULT_CALIBRATION, (frame.shape[1], frame.shape[0]), dictionary
            )
        except SystemExit as exc:
            camera.release()
            print(f"Camera calibration not usable: {exc}")
            return False
        self._camera = camera
        self._detector = vision.make_aruco_detector(dictionary)
        self._calibration = calibration
        self._homography = calibration["pixel_to_robot_xy"]
        self._z_plane = np.asarray(calibration["table_z_plane"], dtype=float)
        self._border = cv2.convexHull(np.asarray(
            [m["image_xy"] for m in calibration["markers"]], dtype=np.float32
        ))
        self._ready = True
        return True

    def locate(self) -> list[tuple[float, float, float]] | None:
        """Robot XYZ (table height) under each yellow block's top center."""
        if not self._ready and not self._open():
            return None
        from util import detect_colored_blocks

        # Drop frames buffered while the camera sat idle.
        for _ in range(5):
            self._camera.grab()
        clusters: list[list[np.ndarray]] = []
        for _ in range(YELLOW_FRAMES):
            ok, frame = self._camera.read()
            if not ok or frame is None:
                print("Camera frame lost.")
                return None
            markers = self._vision.detect_markers(frame, self._detector)
            self._border = self._vision.aruco_border(markers, self._border)
            for block in detect_colored_blocks(frame, border=self._border, border_margin=30):
                xy = np.asarray(self._vision.map_xy(block["contact_xy"], self._homography))
                for cluster in clusters:
                    if np.linalg.norm(np.mean(cluster, axis=0) - xy) <= YELLOW_MATCH_MM:
                        cluster.append(xy)
                        break
                else:
                    clusters.append([xy])
        points = []
        for cluster in clusters:
            if len(cluster) >= YELLOW_MIN_HITS:
                x, y = np.mean(cluster, axis=0)
                z = self._z_plane[0] * x + self._z_plane[1] * y + self._z_plane[2]
                points.append((float(x), float(y), float(z)))
        return sorted(points)

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()


def _save_yellow(camera: YellowCamera) -> None:
    print("Looking for yellow blocks...")
    points = camera.locate()
    if points is None:
        return
    if not points:
        print("No yellow block found inside the marker border.")
        return
    stamp = _timestamp()
    path = _append_rows([(stamp, x, y, z, "", "", "", "camera") for x, y, z in points])
    for index, (x, y, z) in enumerate(points, start=1):
        print(f"Yellow {index} (top center):  X={x:8.2f}   Y={y:8.2f}   "
              f"table Z={z:8.2f}")
    if path is not None:
        print(f"Saved {len(points)} camera point(s) to {path}")


def _set_gripper(rbt: RobotClient, close: bool) -> None:
    action = "Closing" if close else "Opening"
    print(f"{action} gripper...")
    try:
        result = rbt.tool.close(wait=True) if close else rbt.tool.open(wait=True)
    except (MotionError, TimeoutError) as exc:
        print(f"Gripper command failed: {exc}")
        return
    if result < 0:
        print(f"Controller rejected the gripper command (result {result}).")
    else:
        print(f"Gripper {'closed' if close else 'open'}.")


def _solve_free_rx(
    pose: list[float], seed_q: np.ndarray
) -> tuple[float, np.ndarray] | None:
    """Find the reachable Rx nearest ``pose[3]`` with X, Y, Z, Ry and Rz fixed.

    Uses the server's robot model and IK safety margins, so a pose accepted
    here is one the server's servo_l can track. Only solutions on the same IK
    branch as ``seed_q`` count as reachable.
    """
    x, y, z = (value / 1000.0 for value in pose[:3])
    ry, rz = math.radians(pose[4]), math.radians(pose[5])
    for offset in RX_OFFSETS:
        rx = pose[3] + offset
        se3_from_rpy(x, y, z, math.radians(rx), ry, rz, _target_se3)
        result = solve_ik(PAROL6_ROBOT.robot, _target_se3, seed_q, quiet_logging=True)
        if result.success and np.abs(result.q - seed_q).max() <= MAX_JOINT_STEP_RAD:
            return rx, result.q.copy()
    return None


def _clear_cartesian_stream(rbt: RobotClient) -> None:
    """Retire stale Cartesian streaming state before a jog_l starts.

    A new jog_l reuses the server's Cartesian streamer when a STOP or an
    unfinished servo_l left it active, and then fails IK from a stale pose.
    A servo_l to the current pose always resyncs the streamer and finishes
    within a few control ticks.
    """
    pose = rbt.pose(frame="WRF")
    if pose is None:
        return
    rbt.servo_l(pose, speed=JOG_SPEED, accel=JOG_ACCEL)
    # servo_l is fire-and-forget: give the controller a few ticks to pick it
    # up before polling, or an idle reading could arrive first.
    time.sleep(0.03)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        activity = rbt.activity()
        if activity is None or not activity.command:
            return
        time.sleep(0.01)


def main() -> None:
    global JOG_SPEED, JOG_ACCEL, TRANSLATION_FRAME, RX_FREE
    rbt = RobotClient(host=HOST, port=PORT)
    yellow_camera = YellowCamera()
    listener: keyboard.Listener | None = None
    moving = False
    last_active_key: str | None = None
    jog_rejected = False
    last_hold_report = 0.0
    servo_mode = False
    servo_target: list[float] = []
    servo_seed = np.zeros(6)
    reach_reported = False
    try:
        tool_index = rbt.select_tool(TOOL, TOOL_VARIANT)
        if tool_index >= 0:
            rbt.wait_command(tool_index)
        PAROL6_ROBOT.apply_tool(TOOL, TOOL_VARIANT)
        print("Cartesian keyboard control connected.")
        print("Hold one jog key at a time; release it to stop.")
        print("  X: A / D    Y: S / W    Z: Q / E")
        print("  RX: L / J   RY: K / I   RZ: U / O")
        print("  H: home    G: calibrated home (re-reference joints)")
        print("  Space: stop    Esc: stop and quit")
        print("  P: print coordinates    C: append coordinates to CSV")
        print("  Y: find yellow blocks with the camera and append them to CSV")
        print("  V: open gripper    B: close gripper")
        print("  [ / ]: slower / faster    - / =: lower / higher acceleration")
        print("  F: toggle translation frame (WRF / TRF)")
        rx_state = "on" if RX_FREE else "off"
        print(f"  R: toggle Rx free during WRF translation (now {rx_state})")
        _read_pose(rbt)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while True:
            keys, actions, should_quit = _read_state()
            if should_quit:
                break

            if "space" in keys:
                if moving:
                    rbt.stop()
                    moving = False
                    print("Stopped.")
                last_active_key = None
                jog_rejected = False
                time.sleep(1.0 / LOOP_HZ)
                continue

            if "h" in actions or "g" in actions:
                calibrate = "g" in actions
                rbt.stop()
                moving = False
                with state_lock:
                    pressed_keys.difference_update(JOG_KEYS)
                if calibrate:
                    print("Calibrated homing: each joint seeks its limit switch...")
                else:
                    print("Homing...")
                try:
                    rbt.home(wait=True, calibrate=calibrate, timeout=120.0)
                    print("Home complete.")
                except (MotionError, TimeoutError) as exc:
                    print(f"Homing failed: {exc}")
                last_active_key = None
                jog_rejected = False
                continue

            if "p" in actions:
                _read_pose(rbt)
            if "c" in actions:
                _save_pose(rbt)
            if "y" in actions:
                if moving:
                    rbt.stop()
                    moving = False
                with state_lock:
                    pressed_keys.difference_update(JOG_KEYS)
                _save_yellow(yellow_camera)
                last_active_key = None
                continue
            if "v" in actions:
                _set_gripper(rbt, close=False)
            if "b" in actions:
                _set_gripper(rbt, close=True)
            if "[" in actions:
                JOG_SPEED = max(0.05, JOG_SPEED - 0.05)
                print(f"Jog speed: {JOG_SPEED:.2f}")
            if "]" in actions:
                JOG_SPEED = min(1.0, JOG_SPEED + 0.05)
                print(f"Jog speed: {JOG_SPEED:.2f}")
            if "-" in actions:
                JOG_ACCEL = max(0.05, JOG_ACCEL - 0.05)
                print(f"Jog acceleration: {JOG_ACCEL:.2f}")
            if "=" in actions:
                JOG_ACCEL = min(1.0, JOG_ACCEL + 0.05)
                print(f"Jog acceleration: {JOG_ACCEL:.2f}")
            if "f" in actions:
                TRANSLATION_FRAME = "TRF" if TRANSLATION_FRAME == "WRF" else "WRF"
                last_active_key = None
                print(f"Translation frame: {TRANSLATION_FRAME}")
            if "r" in actions:
                RX_FREE = not RX_FREE
                last_active_key = None
                print(f"Rx free: {'on' if RX_FREE else 'off'}")

            active_key = next((key for key in JOG_KEYS if key in keys), None)
            if active_key is None:
                if moving:
                    # Let the jog duration expire instead of sending STOP: a STOP
                    # mid-jog leaves the server's Cartesian streamer latched on a
                    # stale pose, and every later jog then fails IK.
                    moving = False
                    print("Released.")
                last_active_key = None
                jog_rejected = False
            else:
                axis, direction, frame = JOG_KEYS[active_key]
                if axis in TRANSLATION_INDEX:
                    frame = TRANSLATION_FRAME
                if active_key != last_active_key:
                    last_active_key = active_key
                    last_hold_report = time.monotonic()
                    reach_reported = False
                    servo_mode = (
                        RX_FREE and axis in TRANSLATION_INDEX and frame == "WRF"
                    )
                    sign = "+" if direction > 0 else "-"
                    angles = rbt.angles()
                    jog_rejected = angles is not None and not any(angles)
                    if jog_rejected:
                        print(
                            "Robot is not homed (all joint angles read 0). "
                            "Press H first."
                        )
                    elif servo_mode:
                        pose = rbt.pose(frame="WRF")
                        if pose is None or angles is None:
                            jog_rejected = True
                            print("Pose unavailable; cannot start the jog.")
                        else:
                            servo_target = list(pose)
                            servo_seed = np.radians(angles)
                            print(
                                f"Jogging {active_key.upper()}: {axis}{sign} "
                                "in WRF, Rx free."
                            )
                    else:
                        _clear_cartesian_stream(rbt)
                        print(f"Jogging {active_key.upper()}: {axis}{sign} in {frame}.")

                if jog_rejected:
                    pass
                elif servo_mode:
                    index = TRANSLATION_INDEX[axis]
                    candidate = list(servo_target)
                    actual = rbt.pose(frame="WRF")
                    lead = 0.0
                    if actual is not None:
                        lead = (candidate[index] - actual[index]) * direction
                    if lead < MAX_LEAD_MM:
                        step = JOG_SPEED * LINEAR_MAX_MM_S / LOOP_HZ
                        candidate[index] += direction * step
                    solved = _solve_free_rx(candidate, servo_seed)
                    if solved is None:
                        if not reach_reported:
                            print(
                                f"{axis} reach limit: no reachable Rx within "
                                f"±{RX_SEARCH_MAX_DEG:.0f}°."
                            )
                            reach_reported = True
                    else:
                        candidate[3], servo_seed = solved
                        servo_target = candidate
                        rbt.servo_l(
                            servo_target,
                            speed=min(1.0, JOG_SPEED * 1.25),
                            accel=JOG_ACCEL,
                        )
                        moving = True
                        now = time.monotonic()
                        if now - last_hold_report >= 1.0:
                            print(
                                f"{active_key.upper()} is held; "
                                f"Rx now {servo_target[3]:.1f}°."
                            )
                            last_hold_report = now
                else:
                    command_index = rbt.jog_l(
                        frame,
                        axis,
                        JOG_SPEED * direction,
                        duration=JOG_DURATION,
                        accel=JOG_ACCEL,
                    )
                    if command_index < 0:
                        rbt.stop()
                        moving = False
                        jog_rejected = True
                        print(
                            f"Controller rejected {axis} jog (result {command_index}). "
                            "Check the server terminal for its error."
                        )
                    else:
                        moving = True
                        now = time.monotonic()
                        if now - last_hold_report >= 1.0:
                            print(f"{active_key.upper()} is held; jog commands are being sent.")
                            last_hold_report = now

            time.sleep(1.0 / LOOP_HZ)
    finally:
        yellow_camera.close()
        if listener is not None:
            listener.stop()
        try:
            rbt.stop()
        finally:
            rbt.close()
        print("Stopped. Connection closed.")


if __name__ == "__main__":
    main()
