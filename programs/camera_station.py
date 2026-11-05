"""Live camera view with Cartesian jogging, table calibration and seek.

Start ``parol6-server`` in another terminal, then run this from the repo's
``.venv``:

    .venv/bin/python programs/camera_station.py --camera 0

Keys are read from the camera window, so click it first; typing elsewhere
does nothing. Hold one movement key at a time.

    Move       A / D  X    S / W  Y    Q / E  Z
    Rotate     L / J  RX   K / I  RY   U / O  RZ
    Robot      H home   G calibrated home   V open   B close   Space STOP
    Jogging    [ ] speed   - = acceleration   F WRF/TRF   R Rx free
    Save       P print pose   C save pose   Y save yellow blocks to the CSV
    Vision     X start/stop seek (yellow, red)   T T calibrate table
               M M drop calibrate (block in the gripper first)
    Camera     1 auto exposure on/off   2 autofocus on/off
               Z show/hide the green outline
    Quit       Esc

Seek homes the arm, then picks each yellow and red block, dropping yellow ones
at home and red ones beside the base, where the arm waits for the next block
(SEEK_DROPS in OpenCV/seek_yellow.py), until X or Space. Space stops the arm
where it is at once, and nothing moves afterwards until you press a key; after
stopping a move the arm takes about 10 s to accept new commands. Jogging is off while seeking.

T and M need pressing twice so that a stray key can't start them. After the
camera moves, T is all it needs: drop-calibrate pairs carry over through the
four markers. That relies on the markers staying put on the table; if they
moved, the pairs are skipped with a warning until they are back.

All four block colors are shown with their angle on the table, and a stack
with its level. Only yellow and red blocks are picked, and only yellow ones
saved, which is what drop calibrate uses; picks happen only once their
angle has held steady. A block is a 3D cube fitted to its outline
(OpenCV/cube_pose.py), so a tilted view or a lit side doesn't shift it.

The blue box runs from the markers out to the far edge of the green outline.

The green outline surrounds everywhere the gripper reaches the table vertically with at
least 90 degrees of yaw. It is placed through the four markers, so it stays
true when the camera moves. Build it with OpenCV/reach_map.py.

M closes this window while programs/drop_calibrate.py runs, using the exposure
and focus set here, and comes back when it ends.

The Exposure and Focus sliders talk to the C920 over USB, since OpenCV's
camera properties do nothing on macOS. Moving a slider turns that auto mode
off. The camera keeps the settings after this program exits, so
drop_calibrate.py sees the same picture.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

OPENCV_DIR = Path(__file__).resolve().parent.parent / "OpenCV"
sys.path.insert(0, str(OPENCV_DIR))

import parol6.PAROL6_ROBOT as PAROL6_ROBOT  # noqa: E402
from parol6 import RobotClient  # noqa: E402
from parol6.utils.errors import MotionError  # noqa: E402
from pynput import keyboard  # noqa: E402

from camera_controls import FOCUS_STEP, CameraControls, open_controls  # noqa: E402
from cartesian_keyboard import (  # noqa: E402
    JOG_KEYS,
    LINEAR_MAX_MM_S,
    LOOP_HZ,
    JOG_DURATION,
    MAX_LEAD_MM,
    RX_SEARCH_MAX_DEG,
    TRANSLATION_INDEX,
    _append_rows,
    _clear_cartesian_stream,
    _key_name,
    _read_pose,
    _save_pose,
    _solve_free_rx,
    _timestamp,
)
from main import (  # noqa: E402
    CALIBRATION_FRAMES,
    COORDS_CSV,
    DEFAULT_CALIBRATION,
    DICTIONARIES,
    MARKER_ROBOT_COORDS_MM,
    BlockTracker,
    aruco_border,
    calibrate,
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
    load_drop_correction,
    plan_seek_pick,
    set_gripper,
    table_z,
)
from seek_yellow import (  # noqa: E402
    FORGET_AFTER_FRAMES,
    GRIP_SETTLE_S,
    MOVE_SPEED,
    SAFE_HEIGHT_MM,
    SEEK_DROPS,
    SLOW_SPEED,
    VISITED_RADIUS_MM,
)
from reach_map import (  # noqa: E402
    MIN_YAW_DEG,
    border_extension,
    load_reach_mask,
    work_border,
)
from cube_pose import CubeCamera  # noqa: E402
from table_fit import project  # noqa: E402
from workspace import Workspace  # noqa: E402
from util import BLOCK_COLORS, block_yaw, detect_colored_blocks  # noqa: E402

WINDOW = "Camera station"
DROP_CALIBRATE = Path(__file__).with_name("drop_calibrate.py")
EXPOSURE_SLIDER, FOCUS_SLIDER = "Exposure", "Focus"
# Seconds the second press of T or M has to arrive in.
CONFIRM_WINDOW_S = 3.0
# A camera just released by another program can take a moment to deliver.
FIRST_FRAME_ATTEMPTS = 30
MOTION_TIMEOUT_S = 30.0
# How far outside the blue border (px) a block's center may sit and count.
BORDER_MARGIN_PX = 30
HOME_TIMEOUT_S = 120.0
# How often a motion wait wakes to check for Stop.
ABORT_POLL_S = 0.05


class Aborted(Exception):
    pass


class Rejected(Exception):
    pass


ROBOT_ERRORS = (MotionError, TimeoutError, GripperError, Rejected)


class HeldKeys:
    """Which keys are held down in the camera window.

    cv2.waitKey only reports presses, and pynput reports releases but sees
    every app. A key counts as held once waitKey saw it (the window has
    focus) and until pynput sees it go up.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._down: set[str] = set()
        self._held: set[str] = set()

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        name = _key_name(key)
        if name is not None:
            with self._lock:
                self._down.add(name)

    def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        name = _key_name(key)
        if name is not None:
            with self._lock:
                self._down.discard(name)
                self._held.discard(name)

    def window_press(self, name: str) -> bool:
        """Record a press seen by the window; False if it's an auto-repeat."""
        with self._lock:
            if name in self._held:
                return False
            if name in self._down:
                self._held.add(name)
            return True

    def active_jog_key(self) -> str | None:
        with self._lock:
            return next((key for key in JOG_KEYS if key in self._held), None)

    def drop(self) -> None:
        """Forget held keys, so a jog needs a fresh press."""
        with self._lock:
            self._down.clear()
            self._held.clear()


class Jogger:
    """The streaming half of cartesian_keyboard: one held key at a time."""

    def __init__(self, rbt: RobotClient) -> None:
        self.rbt = rbt
        self.speed = 0.15
        self.accel = 0.30
        self.translation_frame = "WRF"
        self.rx_free = False
        self.restart()
        self.moving = False

    def restart(self) -> None:
        self.last_key: str | None = None
        self.rejected = False

    def halt(self) -> None:
        if self.moving:
            self.rbt.stop()
            self.moving = False
        self.restart()

    def update(self, key: str | None) -> None:
        if key is None:
            # Let the jog duration expire instead of sending STOP: see
            # cartesian_keyboard for why a STOP mid-jog is avoided.
            self.moving = False
            self.restart()
            return
        axis, direction, frame = JOG_KEYS[key]
        if axis in TRANSLATION_INDEX:
            frame = self.translation_frame
        if key != self.last_key:
            self._begin(key, axis, direction, frame)
        if self.rejected:
            return
        if self.servo_mode:
            self._servo_step(axis, direction)
        else:
            self._jog_step(axis, direction, frame)

    def _begin(self, key: str, axis: str, direction: int, frame: str) -> None:
        self.last_key = key
        self.reach_reported = False
        self.servo_mode = self.rx_free and axis in TRANSLATION_INDEX and frame == "WRF"
        angles = self.rbt.angles()
        self.rejected = angles is not None and not any(angles)
        if self.rejected:
            print("Robot is not homed (all joint angles read 0). Press H first.")
        elif self.servo_mode:
            pose = self.rbt.pose(frame="WRF")
            if pose is None or angles is None:
                self.rejected = True
                print("Pose unavailable; cannot start the jog.")
            else:
                self.servo_target = list(pose)
                self.servo_seed = np.radians(angles)
        else:
            _clear_cartesian_stream(self.rbt)

    def _jog_step(self, axis: str, direction: int, frame: str) -> None:
        index = self.rbt.jog_l(frame, axis, self.speed * direction,
                               duration=JOG_DURATION, accel=self.accel)
        if index < 0:
            self.rbt.stop()
            self.moving = False
            self.rejected = True
            print(f"Controller rejected {axis} jog (result {index}). "
                  "Check the server terminal for its error.")
        else:
            self.moving = True

    def _servo_step(self, axis: str, direction: int) -> None:
        index = TRANSLATION_INDEX[axis]
        candidate = list(self.servo_target)
        actual = self.rbt.pose(frame="WRF")
        lead = 0.0 if actual is None else (candidate[index] - actual[index]) * direction
        if lead < MAX_LEAD_MM:
            candidate[index] += direction * self.speed * LINEAR_MAX_MM_S / LOOP_HZ
        solved = _solve_free_rx(candidate, self.servo_seed)
        if solved is None:
            if not self.reach_reported:
                print(f"{axis} reach limit: no reachable Rx within "
                      f"±{RX_SEARCH_MAX_DEG:.0f}°.")
                self.reach_reported = True
            return
        candidate[3], self.servo_seed = solved
        self.servo_target = candidate
        self.rbt.servo_l(candidate, speed=min(1.0, self.speed * 1.25), accel=self.accel)
        self.moving = True


class RobotWorker(threading.Thread):
    """Owns the arm's motion, so slow frames can't stutter a jog and a long
    move can't freeze the camera view."""

    def __init__(self, rbt: RobotClient, held: HeldKeys) -> None:
        super().__init__(daemon=True)
        self.rbt = rbt
        self.held = held
        self.jogger = Jogger(rbt)
        self.closing = threading.Event()
        # Seeking locks out jogging, and a failed job ends it.
        self.seeking = False
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self._job: Callable[[], None] | None = None
        self._running = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running or self._job is not None

    def submit(self, job: Callable[[], None]) -> bool:
        with self._lock:
            if self._running or self._job is not None:
                print("The arm is busy; press Space to stop it first.")
                return False
            self._abort.clear()
            self._job = job
            return True

    @property
    def stopping(self) -> bool:
        return self._abort.is_set() and self.busy

    def enforce_stop(self) -> None:
        """Repeat STOP until the job has ended.

        Sending a move blocks until the server answers, so a job can't see
        Stop while sending. Repeating it catches a move that went out just
        after the first STOP, which nothing else could cancel. The answer to
        a stopped move takes about 10 s, and the job holds the arm until then.
        """
        if self.stopping:
            self.rbt.stop()

    def stop_motion(self) -> None:
        with self._lock:
            self._job = None
        self._abort.set()
        self.held.drop()
        self.rbt.stop()

    def _take_job(self) -> Callable[[], None] | None:
        with self._lock:
            job, self._job = self._job, None
            self._running = job is not None
            return job

    def run(self) -> None:
        while not self.closing.is_set():
            job = self._take_job()
            try:
                if job is not None:
                    self.jogger.halt()
                    job()
                else:
                    self.jogger.update(None if self.seeking else self.held.active_jog_key())
            except Aborted:
                print("Stopped. The arm is holding where it is.")
            except ROBOT_ERRORS as exc:
                print(f"Robot error: {exc}")
                self.seeking = False
                self.jogger.restart()
            except RuntimeError:
                # Quitting closes the client under a send that is still blocked.
                if not self.closing.is_set():
                    raise
            finally:
                with self._lock:
                    self._running = False
            if job is None:
                time.sleep(1.0 / LOOP_HZ)

    def run_motion(self, send: Callable[[], int],
                   timeout: float = MOTION_TIMEOUT_S) -> None:
        """Send a motion and wait for it, or raise once Stop is pressed.

        A move's own wait=True is no use here: it ignores a timed-out wait,
        so a stopped move would let the caller go on to its next step. A
        stopped command can also read as completed, so Stop is checked
        whichever way the wait ends.
        """
        self.check_abort()
        index = send()
        if index < 0:
            raise Rejected(f"controller rejected the command (result {index})")
        deadline = time.monotonic() + timeout
        while True:
            finished = self.rbt.wait_command(index, timeout=ABORT_POLL_S)
            self.check_abort()
            if finished:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"motion timed out after {timeout:.0f} s")

    def check_abort(self) -> None:
        if self._abort.is_set():
            raise Aborted

    def gripper(self, close: bool) -> None:
        self.check_abort()
        set_gripper(self.rbt, close=close)

    def home(self, calibrate_joints: bool = False) -> None:
        print("Homing...")
        self.run_motion(lambda: self.rbt.home(calibrate=calibrate_joints), HOME_TIMEOUT_S)
        print("Home complete.")

    def seek_start(self) -> None:
        self.home()
        self.gripper(close=False)
        print(f"Looking for {' and '.join(SEEK_DROPS)} blocks.")

    def pick_and_drop(self, x: float, y: float, z: float, plan: Plan, color: str) -> None:
        """seek_yellow's cycle, except that Stop ends it without any recovery
        move, which would otherwise carry on driving the arm."""
        rbt = self.rbt
        print(f"Picking {color} at X={x:.1f} Y={y:.1f} Z={z:.1f} mm  {plan.describe()}")
        above = None
        try:
            self.run_motion(lambda: rbt.move_j(plan.angles, speed=MOVE_SPEED))
            above = rbt.pose(frame="WRF")
            if above is None:
                raise TimeoutError("pose unavailable above the block")
            self.run_motion(lambda: rbt.move_l([x, y, z, *above[3:]], speed=SLOW_SPEED))
            if self._abort.wait(GRIP_SETTLE_S):
                self.check_abort()
            self.gripper(close=True)
            self.run_motion(lambda: rbt.move_l(above, speed=SLOW_SPEED))
            drop = SEEK_DROPS[color]
            if drop is None:
                self.home()
                self.gripper(close=False)
            else:
                self.run_motion(lambda: rbt.move_j(pose=drop, speed=MOVE_SPEED))
                self.gripper(close=False)
            print("Dropped. Looking for another block...")
        except ROBOT_ERRORS as exc:
            self.check_abort()
            print(f"Pick failed ({exc}); returning home.")
            if above is not None:
                self.run_motion(lambda: rbt.move_l(above, speed=SLOW_SPEED))
            self.home()
            self.gripper(close=False)


class Station:
    def __init__(self, args: argparse.Namespace, camera: cv2.VideoCapture,
                 frame: np.ndarray, rbt: RobotClient, worker: RobotWorker,
                 held: HeldKeys, controls: CameraControls | None) -> None:
        self.args = args
        self.camera = camera
        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.rbt = rbt
        self.worker = worker
        self.held = held
        self.controls = controls
        self.detector = make_aruco_detector(args.dictionary)
        self.tracker = BlockTracker(args.confirm_frames,
                                    yaw_of=lambda block: block_yaw(block, self.to_table))
        self.calibration: dict | None = None
        self.correction = None
        self.border: np.ndarray | None = None
        self.cube_camera: CubeCamera | None = None
        self.workspace = Workspace()
        self.border_extension: np.ndarray | None = None
        self.shift_warned = False
        self.blocks: list[dict] = []
        self.visited: dict[tuple[float, float], int] = {}  # spot -> frames since seen
        self.unreachable: list[tuple[float, float]] = []
        self.confirm_key: str | None = None
        self.confirm_until = 0.0
        self.confirm_prompt = ""
        self.drop_requested = False
        self.calibration_samples: dict[int, list[np.ndarray]] | None = None
        self.show_reach = True
        self.reach_mask: np.ndarray | None = None
        self.load_vision()
        self.open_window()

    def open_window(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        controls = self.controls
        if controls is not None:
            self.exposure_pos = controls.exposure
            self.focus_pos = controls.focus
            self.auto_exposure = controls.auto_exposure
            self.auto_focus = controls.auto_focus
            for name, (low, high), position in (
                (EXPOSURE_SLIDER, controls.exposure_range, self.exposure_pos),
                (FOCUS_SLIDER, controls.focus_range, self.focus_pos),
            ):
                cv2.createTrackbar(name, WINDOW, position, high, lambda _: None)
                cv2.setTrackbarMin(name, WINDOW, low)
                cv2.setTrackbarPos(name, WINDOW, position)

    def load_vision(self) -> None:
        """Load the calibration and correction; without them the view and
        jogging still work, so T can fix a bad calibration from here."""
        self.calibration = self.correction = self.reach_mask = self.cube_camera = None
        try:
            self.calibration = load_calibration(
                self.args.calibration, self.frame_size, self.args.dictionary
            )
            self.border = cv2.convexHull(np.asarray(
                [m["image_xy"] for m in self.calibration["markers"]], dtype=np.float32
            ))
            self.cube_camera = CubeCamera(self.calibration["pixel_to_robot_xy"], self.frame_size)
            self.border_extension = border_extension(self.calibration)
            self.border = work_border(self.border, self.border_extension)
            self.reach_mask = load_reach_mask(
                self.calibration["pixel_to_robot_xy"], self.frame_size
            )
            if self.reach_mask is not None:
                self.reach_outline, _ = cv2.findContours(
                    self.reach_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
            self.correction = load_drop_correction(
                self.args.calibration, COORDS_CSV, self.args.drop_pairs
            )
            print(self.correction.summary())
        except SystemExit as exc:
            print(exc)

    def to_table(self, pixels: np.ndarray) -> np.ndarray:
        xy = project(self.calibration["pixel_to_robot_xy"], pixels)
        return self.correction.to_robot_points(xy) if self.correction else xy

    def to_robot(self, pixel_xy: tuple[float, float],
                 height: float = 0.0) -> tuple[float, float]:
        xy = map_xy(pixel_xy, self.calibration["pixel_to_robot_xy"])
        return self.correction.to_robot(xy, height) if self.correction else xy

    def process(self, frame: np.ndarray) -> np.ndarray:
        detections = detect_markers(frame, self.detector)
        view = frame.copy()
        self.draw_reach(view)
        marker_overlay(view, detections)
        self.sample_calibration(detections)
        busy = self.worker.busy
        target = None
        detected = [] if busy else detect_colored_blocks(frame, colors=BLOCK_COLORS,
                                                         camera=self.cube_camera)
        if self.calibration is not None:
            self.shift_warned = check_camera_shift(detections, self.calibration,
                                                   self.shift_warned)
            self.border = work_border(aruco_border(detections, self.border),
                                      self.border_extension)
            cv2.polylines(view, [np.rint(self.border).astype(np.int32)], True,
                          (255, 160, 0), 2)
            target = self.track_blocks(detected, view, busy)
        if not busy:
            self.draw_ignored_blocks(detected, view)
        if self.worker.seeking and target is not None and not busy:
            self.go_pick(target)
        self.draw_hud(view, busy, len(detections))
        return view

    def draw_reach(self, view: np.ndarray) -> None:
        if not self.show_reach or self.reach_mask is None:
            return
        cv2.polylines(view, self.reach_outline, True, (60, 255, 60), 2)

    def inside_border(self, block: dict) -> bool:
        border = np.asarray(self.border, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(border, block["center_xy"], True) >= -BORDER_MARGIN_PX

    @property
    def yellow_blocks(self) -> list[dict]:
        return [b for b in self.blocks if b["color"] == "yellow"]

    def track_blocks(self, detected: list[dict], view: np.ndarray,
                     busy: bool) -> tuple[float, float, float, float, str] | None:
        # The arm passes through the view while moving; restart confirmation
        # once it stops so its motion can't trigger a visit.
        if busy:
            self.tracker.tracks.clear()
        raw = [b for b in detected if self.inside_border(b)]
        self.blocks = self.tracker.update(raw)
        if not busy:
            seen = [self.to_robot(b["contact_xy"]) for b in raw if b["color"] in SEEK_DROPS]
            for spot in list(self.visited):
                if any(math.dist(spot, s) <= VISITED_RADIUS_MM for s in seen):
                    self.visited[spot] = 0
                elif self.visited[spot] >= FORGET_AFTER_FRAMES:
                    del self.visited[spot]
                else:
                    self.visited[spot] += 1
        target, leftmost = None, float("inf")
        tops = []
        for block in self.blocks:
            steady = block["steady_contact_xy"]
            height = block.get("height_mm", 0.0)
            x, y = self.to_robot(steady or block["contact_xy"], height)
            sought = block["color"] in SEEK_DROPS
            done = sought and any(math.dist((x, y), p) <= VISITED_RADIUS_MM
                                  for p in [*self.visited, *self.unreachable])
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
            yaw = block["steady_yaw"]
            angle = "?" if yaw is None else f"{yaw:.0f} deg"
            level = block.get("level", 0)
            if not steady:
                text = f"{block['color']} no good fit"
            else:
                text = f"{block['color']} {angle}{f' level {level}' if level else ''}"
                tops.append({"x": x, "y": y, "level": level, "yaw": yaw or 0.0,
                             "color": block["color"]})
            cv2.putText(view, f"{text} X{x:.0f} Y{y:.0f}", (left, max(20, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            # Leftmost in the camera first, so seek sweeps the table left to right.
            if (sought and steady and block["yaw_ready"] and not done
                    and block["center_xy"][0] < leftmost):
                target, leftmost = (x, y, yaw, height, block["color"]), block["center_xy"][0]
        self.workspace.update(tops)
        return target

    def draw_ignored_blocks(self, detected: list[dict], view: np.ndarray) -> None:
        """Show blocks that detection sees but tracking won't use, and why."""
        for block in detected:
            if self.calibration is None:
                reason = "no calibration"
            elif not self.inside_border(block):
                reason = "outside the blue box"
            else:
                continue
            cv2.polylines(view, [np.rint(block["box"]).astype(np.int32)], True,
                          (128, 128, 128), 2)
            left, top, _, _ = block["bbox"]
            cv2.putText(view, f"ignored: {reason}", (left, max(20, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def go_pick(self, target: tuple[float, float, float, float, str]) -> None:
        x, y, yaw, height, color = target
        z = self.correction.pick_z(x, y, height)
        worker = self.worker

        def job() -> None:
            # Planning is a long IK search, so it stays off the camera loop.
            planned = plan_seek_pick(self.correction, x, y, z, SAFE_HEIGHT_MM, yaw)
            if planned is None:
                print(f"Block at X={x:.1f} Y={y:.1f} can't be gripped at "
                      f"{yaw:.0f} deg from here; skipping.")
                self.unreachable.append((x, y))
            else:
                worker.pick_and_drop(planned[0], planned[1], z, planned[2], color)

        if worker.submit(job):
            self.visited[(x, y)] = 0

    def toggle_seek(self) -> None:
        if self.worker.seeking:
            self.worker.seeking = False
            print("Seek off; finishing the current pick.")
            return
        self.load_vision()
        if self.calibration is None or self.correction is None:
            print("Seek needs the problem above fixed first.")
            return
        self.visited.clear()
        self.unreachable.clear()
        self.tracker.tracks.clear()
        if self.worker.submit(self.worker.seek_start):
            self.worker.seeking = True

    def press_calibrate(self) -> None:
        if self.worker.seeking or self.worker.busy:
            print("Stop the arm before calibrating.")
        elif self.confirmed("t", "PRESS T AGAIN to recalibrate the table"):
            self.calibration_samples = {i: [] for i in MARKER_ROBOT_COORDS_MM}

    def press_drop_calibrate(self) -> None:
        if self.worker.seeking or self.worker.busy:
            print("Stop the arm before drop calibrating.")
        elif self.calibration is None:
            print("Drop calibrate needs a table calibration first: press T twice.")
        elif self.confirmed("m", "PRESS M AGAIN to drop calibrate (block in the gripper)"):
            self.drop_requested = True

    def confirmed(self, key: str, prompt: str) -> bool:
        """True on the second press of ``key`` inside the confirm window."""
        now = time.monotonic()
        if self.confirm_key == key and now < self.confirm_until:
            self.confirm_until = 0.0
            return True
        self.confirm_key, self.confirm_prompt = key, prompt
        self.confirm_until = now + CONFIRM_WINDOW_S
        print(prompt)
        return False

    def camera_setting_args(self) -> list[str]:
        """This view's manual settings, for a program taking over the camera."""
        if self.controls is None:
            return []
        args = [] if self.auto_exposure else ["--exposure", str(self.exposure_pos)]
        return args + ([] if self.auto_focus else ["--focus", str(self.focus_pos)])

    def restore_camera_settings(self) -> None:
        if self.controls is None:
            return
        if not self.auto_exposure:
            self.controls.set_exposure(self.exposure_pos)
        if not self.auto_focus:
            self.controls.set_focus(self.focus_pos)

    def run_drop_calibrate(self) -> None:
        # It needs the camera and has its own window and key loop.
        self.drop_requested = False
        self.held.drop()
        self.camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        subprocess.run([sys.executable, str(DROP_CALIBRATE), "--camera",
                        str(self.args.camera), *self.camera_setting_args()], check=False)
        self.held.drop()
        self.camera, _ = open_camera(self.args)
        self.restore_camera_settings()
        self.open_window()
        self.tracker.tracks.clear()
        self.load_vision()

    def sample_calibration(self, detections: list[dict]) -> None:
        samples = self.calibration_samples
        if samples is None:
            return
        visible = {m["id"]: m["corners"] for m in detections if m["id"] in samples}
        if len(visible) < len(samples):
            return
        for marker_id, corners in visible.items():
            samples[marker_id].append(corners)
        if len(samples[marker_id]) < CALIBRATION_FRAMES:
            return
        self.calibration_samples = None
        averaged = {i: np.mean(c, axis=0) for i, c in samples.items()}
        try:
            calibrate(self.frame_size, averaged, self.args.calibration,
                      self.args.dictionary, self.args.marker_size)
        except ValueError as exc:
            print(f"Calibration failed: {exc}")
            return
        self.shift_warned = False
        self.load_vision()

    def save_yellow(self) -> None:
        yellow = self.yellow_blocks
        if self.calibration is None or not yellow:
            print("No yellow block found inside the blue box.")
            return
        homography = self.calibration["pixel_to_robot_xy"]
        z_plane = np.asarray(self.calibration["table_z_plane"], dtype=float)
        points = [map_xy(b["steady_contact_xy"] or b["contact_xy"], homography)
                  for b in yellow]
        stamp = _timestamp()
        path = _append_rows([(stamp, x, y, table_z(z_plane, x, y), "", "", "", "camera")
                             for x, y in points])
        if path is not None:
            print(f"Saved {len(points)} camera point(s) to {path}")

    def draw_hud(self, view: np.ndarray, busy: bool, marker_count: int) -> None:
        now = time.monotonic()
        if self.calibration_samples is not None:
            done = max(len(c) for c in self.calibration_samples.values())
            status = f"CALIBRATING {done}/{CALIBRATION_FRAMES}: all 4 markers must be clear"
        elif self.worker.stopping:
            status = "STOPPED: arm holding, ready again in about 10 s"
        elif now < self.confirm_until:
            status = self.confirm_prompt
        elif self.worker.seeking:
            status = f"SEEK {'moving' if busy else 'searching'}"
        else:
            status = "ARM BUSY" if busy else "JOG"
        jogger = self.worker.jogger
        lines = [
            status,
            f"Speed {jogger.speed:.2f}  accel {jogger.accel:.2f}  "
            f"{jogger.translation_frame}  Rx free {'on' if jogger.rx_free else 'off'}  "
            f"markers {marker_count}/{len(MARKER_ROBOT_COORDS_MM)}  "
            + "  ".join(f"{c} {sum(b['color'] == c for b in self.blocks)}" for c in BLOCK_COLORS)
            + f"  towers {len(self.workspace.towers())}",
        ]
        if self.controls is not None:
            lines.append(self.camera_status())
        if self.calibration is None:
            lines.append("No usable calibration: see the terminal")
        elif self.correction is None:
            lines.append("No drop-calibrate pairs: run programs/drop_calibrate.py")
        if self.show_reach and self.reach_mask is not None:
            lines.append(f"Inside green: vertical gripper, {MIN_YAW_DEG} deg yaw (Z hides)")
        lines.append("X seek  T T calibrate  M M drop calibrate  H home  Space STOP  Esc quit")
        for row, text in enumerate(lines):
            scale = 0.8 if row == 0 else 0.6
            cv2.putText(view, text, (18, 32 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), 4)
            cv2.putText(view, text, (18, 32 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 255, 255), 2)

    def camera_status(self) -> str:
        exposure = "auto" if self.auto_exposure else str(self.exposure_pos)
        focus = "auto" if self.auto_focus else str(self.focus_pos)
        return f"Exposure {exposure} (1)   Focus {focus} (2)"

    def read_sliders(self) -> None:
        controls = self.controls
        if controls is None:
            return
        exposure = cv2.getTrackbarPos(EXPOSURE_SLIDER, WINDOW)
        if exposure != self.exposure_pos:
            self.exposure_pos = exposure
            self.auto_exposure = False
            controls.set_exposure(exposure)
        focus = cv2.getTrackbarPos(FOCUS_SLIDER, WINDOW)
        focus = FOCUS_STEP * round(focus / FOCUS_STEP)
        if focus != self.focus_pos:
            self.focus_pos = focus
            self.auto_focus = False
            controls.set_focus(focus)

    def toggle_auto_exposure(self) -> None:
        controls = self.controls
        if controls is None:
            print("Camera controls are unavailable; see the message at startup.")
        elif self.auto_exposure:
            # Fix the value auto chose, so the picture doesn't jump.
            self.exposure_pos = controls.exposure
            self.auto_exposure = False
            controls.set_exposure(self.exposure_pos)
            cv2.setTrackbarPos(EXPOSURE_SLIDER, WINDOW, self.exposure_pos)
        else:
            self.auto_exposure = True
            controls.set_auto_exposure()

    def toggle_auto_focus(self) -> None:
        controls = self.controls
        if controls is None:
            print("Camera controls are unavailable; see the message at startup.")
        elif self.auto_focus:
            self.focus_pos = controls.focus
            self.auto_focus = False
            controls.set_focus(self.focus_pos)
            cv2.setTrackbarPos(FOCUS_SLIDER, WINDOW, self.focus_pos)
        else:
            self.auto_focus = True
            controls.set_auto_focus()

    def handle_key(self, code: int) -> bool:
        """Act on a key from the window; False means quit."""
        if code == 27:
            return False
        name = "space" if code == 32 else chr(code).lower()
        worker = self.worker
        jogger = worker.jogger
        if not self.held.window_press(name) or name in JOG_KEYS:
            return True
        if name == "space":
            worker.seeking = False
            self.calibration_samples = None
            worker.stop_motion()
            print("STOP.")
        elif name == "x":
            self.toggle_seek()
        elif name == "t":
            self.press_calibrate()
        elif name == "m":
            self.press_drop_calibrate()
        elif name == "1":
            self.toggle_auto_exposure()
        elif name == "2":
            self.toggle_auto_focus()
        elif name == "z":
            self.show_reach = not self.show_reach
        elif name == "p":
            _read_pose(self.rbt)
        elif name == "c":
            _save_pose(self.rbt)
        elif name == "y":
            self.save_yellow()
        elif name == "[":
            jogger.speed = max(0.05, jogger.speed - 0.05)
        elif name == "]":
            jogger.speed = min(1.0, jogger.speed + 0.05)
        elif name == "-":
            jogger.accel = max(0.05, jogger.accel - 0.05)
        elif name == "=":
            jogger.accel = min(1.0, jogger.accel + 0.05)
        elif name == "f":
            jogger.translation_frame = "TRF" if jogger.translation_frame == "WRF" else "WRF"
            jogger.restart()
        elif name == "r":
            jogger.rx_free = not jogger.rx_free
            jogger.restart()
        elif self.worker.seeking and name in "hgvb":
            print("Press X or Space to stop seeking first.")
        elif name == "h":
            worker.submit(worker.home)
        elif name == "g":
            worker.submit(lambda: worker.home(calibrate_joints=True))
        elif name == "v":
            worker.submit(lambda: worker.gripper(close=False))
        elif name == "b":
            worker.submit(lambda: worker.gripper(close=True))
        return True

    def run(self) -> None:
        while True:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                print("Camera frame lost.")
                return
            self.worker.enforce_stop()
            cv2.imshow(WINDOW, self.process(frame))
            code = cv2.waitKey(1) & 0xFF
            if code != 0xFF and not self.handle_key(code):
                return
            if self.drop_requested:
                self.run_drop_calibrate()
                continue
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            self.read_sliders()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dictionary", choices=DICTIONARIES, default="4X4_50")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--marker-size", type=float, default=None,
                        help="Printed ArUco marker side length in mm; "
                             "measured from the touched points if omitted")
    parser.add_argument("--confirm-frames", type=int, default=10,
                        help="Frames a block must be seen before the robot goes to it")
    parser.add_argument("--drop-pairs", type=int, default=None,
                        help="Use only the most recent N drop-calibrate pairs (default: all)")
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> tuple[cv2.VideoCapture, np.ndarray]:
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    for _ in range(FIRST_FRAME_ATTEMPTS):
        ok, frame = camera.read()
        if ok and frame is not None:
            return camera, frame
        time.sleep(0.1)
    camera.release()
    raise SystemExit("Camera opened but did not provide a frame.")


def main() -> None:
    args = parse_args()
    PAROL6_ROBOT.apply_tool(TOOL, TOOL_VARIANT)
    camera, frame = open_camera(args)

    controls = open_controls()
    held = HeldKeys()
    listener = keyboard.Listener(on_press=held.on_press, on_release=held.on_release)
    rbt = RobotClient(host=HOST, port=PORT)
    worker = RobotWorker(rbt, held)
    station = None
    try:
        tool_index = rbt.select_tool(TOOL, TOOL_VARIANT)
        if tool_index >= 0:
            rbt.wait_command(tool_index)
        station = Station(args, camera, frame, rbt, worker, held, controls)
        listener.start()
        listener.wait()
        if not listener.IS_TRUSTED:
            print("Jogging needs Accessibility permission for this terminal app "
                  "(System Settings > Privacy & Security); other keys still work.")
        worker.start()
        print(__doc__)
        station.run()
    finally:
        worker.closing.set()
        listener.stop()
        if worker.is_alive():
            worker.stop_motion()
            worker.join(timeout=2.0)
        (station.camera if station else camera).release()
        cv2.destroyAllWindows()
        if controls is not None:
            controls.close()
        try:
            rbt.stop()
        finally:
            rbt.close()
        print("Stopped. Connection closed.")


if __name__ == "__main__":
    main()
