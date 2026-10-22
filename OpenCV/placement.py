"""Gripper orientation planning and the drop-calibrate correction.

Shared by ``programs/drop_calibrate.py`` (which places a block on a grid and
logs where the camera saw it) and ``seek_yellow.py`` (which picks blocks up),
so both reach the table with the same gripper orientations.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6 import RobotClient
from parol6.utils.ik import solve_ik
from pinokin import so3_rpy

HOST, PORT = "127.0.0.1", 5001
TOOL, TOOL_VARIANT = "PNEUMATIC", "vertical"
HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]

# Gripper vertical, as when the calibration markers were touched: pose
# readout (Rx, Ry, Rz) = (90, 0, 90).
TOUCH_ROTATION = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

# Let the jaws finish opening/closing before the arm moves, so they don't
# drag the block.
GRIPPER_SETTLE_S = 0.7

# In the pose readout (Rx, Ry, Rz) = (90, heading, 90) is the gripper
# vertical, turned to that heading about the vertical axis.
HEADING_STEP_DEG = 5
# Every point must allow at least this range of Ry at Rx = 90; where it
# doesn't, Rx is lowered in RX_STEP_DEG steps until it does.
MIN_RY_FREEDOM_DEG = 45
RX_STEP_DEG = 5
MAX_RX_DECREASE_DEG = 60
# The descent is checked to this far below the table, so the block can still
# be pressed down where the table height is slightly off.
BELOW_TABLE_MM = 5.0
DESCENT_STEP_MM = 1.0
# A bigger joint jump between descent steps means IK switched branches,
# which a straight move_l can't follow.
MAX_JOINT_STEP_RAD = math.radians(10.0)

# The correction needs at least this many drop-calibrate pairs.
MIN_DROP_PAIRS = 6
# Thin-plate spline smoothing (normalized units) tried by leave-one-out; the
# largest is effectively a plain affine fit.
SMOOTHING_CHOICES = (0.0, 0.001, 0.01, 0.1, 1.0, 1e6)


class GripperError(Exception):
    pass


def set_gripper(rbt: RobotClient, close: bool) -> None:
    result = rbt.tool.close(wait=True) if close else rbt.tool.open(wait=True)
    if result < 0:
        raise GripperError(f"controller rejected gripper {'close' if close else 'open'}")
    time.sleep(GRIPPER_SETTLE_S)


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def table_z(z_plane: np.ndarray, x: float, y: float) -> float:
    return float(z_plane[0] * x + z_plane[1] * y + z_plane[2])


@dataclass
class Plan:
    angles: list[float]  # joint angles (deg) at the safe height
    rotation: np.ndarray
    rx_decrease: int
    ry_range: tuple[float, float]  # reachable heading window, deg

    def describe(self) -> str:
        rpy = np.zeros(3)
        so3_rpy(self.rotation, rpy)
        rx, ry, rz = np.degrees(rpy)
        low, high = self.ry_range
        return (f"Rx={rx:6.1f} Ry={ry:6.1f} Rz={rz:6.1f}  (Ry free over "
                f"{high - low:.0f} deg, Rx lowered {self.rx_decrease} from 90)")


def _ik_seeds(x: float, y: float) -> list[np.ndarray]:
    toward = math.degrees(math.atan2(y, x))
    return [np.radians(HOME_ANGLES)] + [
        np.radians([toward, -90, 180, 0, j5, j6])
        for j5 in (0, -60, 60) for j6 in (180, 90, 270)
    ]


def _descent_start(x: float, y: float, top_z: float, bottom_z: float,
                   rotation: np.ndarray) -> np.ndarray | None:
    """Joint angles (rad) at ``top_z`` from which a straight vertical move
    down to ``bottom_z`` stays reachable on one IK branch, or None."""
    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = (x / 1000.0, y / 1000.0, top_z / 1000.0)
    for seed in _ik_seeds(x, y):
        result = solve_ik(PAROL6_ROBOT.robot, target, seed, quiet_logging=True)
        if not result.success:
            continue
        top = q = result.q.copy()
        for z in np.arange(top_z - DESCENT_STEP_MM, bottom_z - 1e-6, -DESCENT_STEP_MM):
            target[2, 3] = z / 1000.0
            result = solve_ik(PAROL6_ROBOT.robot, target, q, quiet_logging=True)
            if not result.success or np.abs(result.q - q).max() > MAX_JOINT_STEP_RAD:
                break
            q = result.q.copy()
        else:
            return top
        target[2, 3] = top_z / 1000.0
    return None


def _orientation(rx_decrease: float, heading: float) -> np.ndarray:
    # Pose readout Rx = 90 - rx_decrease, Ry = heading, Rz = 90.
    return rot_x(-rx_decrease) @ rot_z(heading) @ TOUCH_ROTATION


def plan_point(x: float, y: float, table: float, safe_height: float) -> Plan | None:
    """Orientation for reaching the table at (x, y).

    ``table`` is the TCP height at the table there. Uses Rx = 90 if some
    window of at least MIN_RY_FREEDOM_DEG of headings (Ry) all reach it;
    otherwise lowers Rx step by step until one does. Uses the middle of the
    widest such window. Needs ``PAROL6_ROBOT.apply_tool`` called first.
    """
    top, bottom = table + safe_height, table - BELOW_TABLE_MM
    headings = list(range(-180, 180, HEADING_STEP_DEG))
    for rx_decrease in range(0, MAX_RX_DECREASE_DEG + 1, RX_STEP_DEG):
        starts = {h: _descent_start(x, y, top, bottom, _orientation(rx_decrease, h))
                  for h in headings}
        ok = [starts[h] is not None for h in headings]
        if all(ok):
            run_start, run_length = 0, len(headings)
        else:
            # Longest circular run of reachable headings.
            run_start = run_length = 0
            first_gap = ok.index(False)
            length = 0
            for i in range(first_gap + 1, first_gap + 1 + len(headings)):
                if ok[i % len(headings)]:
                    length += 1
                    if length > run_length:
                        run_start, run_length = i - length + 1, length
                else:
                    length = 0
        span = (run_length - 1) * HEADING_STEP_DEG
        if run_length and span >= MIN_RY_FREEDOM_DEG:
            middle = headings[(run_start + run_length // 2) % len(headings)]
            low = headings[run_start % len(headings)]
            return Plan(np.degrees(starts[middle]).tolist(),
                        _orientation(rx_decrease, middle), rx_decrease,
                        (low, low + span))
    return None


def _tps_kernel(sq_dist: np.ndarray) -> np.ndarray:
    # r^2 log r, written in terms of r^2; 0 at r = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 0.5 * sq_dist * np.log(sq_dist)
    return np.nan_to_num(k)


class _ThinPlate:
    """Smoothing thin-plate spline from 2-D points to vector values."""

    def __init__(self, src: np.ndarray, values: np.ndarray, smoothing: float):
        self.center = src.mean(axis=0)
        self.scale = float(np.linalg.norm(src - self.center, axis=1).mean()) or 1.0
        self.src = (src - self.center) / self.scale
        n = len(src)
        kernel = _tps_kernel(((self.src[:, None] - self.src[None]) ** 2).sum(-1))
        poly = np.hstack([np.ones((n, 1)), self.src])
        system = np.block([[kernel + smoothing * np.eye(n), poly],
                           [poly.T, np.zeros((3, 3))]])
        rhs = np.vstack([values, np.zeros((3, values.shape[1]))])
        solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
        self.weights, self.affine = solution[:n], solution[n:]

    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = (np.atleast_2d(points) - self.center) / self.scale
        kernel = _tps_kernel(((pts[:, None] - self.src[None]) ** 2).sum(-1))
        return kernel @ self.weights + np.hstack([np.ones((len(pts), 1)), pts]) @ self.affine


def _fit_by_leave_one_out(src: np.ndarray, values: np.ndarray) -> tuple[_ThinPlate, float]:
    """Spline with the smoothing that predicts left-out pairs best, and that
    mean left-out error."""
    best = None
    for smoothing in SMOOTHING_CHOICES:
        errors = []
        for i in range(len(src)):
            keep = np.arange(len(src)) != i
            fit = _ThinPlate(src[keep], values[keep], smoothing)
            errors.append(float(np.linalg.norm(fit(src[i])[0] - values[i])))
        error = float(np.mean(errors))
        if best is None or error < best[1]:
            best = (smoothing, error)
    return _ThinPlate(src, values, best[0]), best[1]


class DropCorrection:
    """Camera XY -> robot XY and table pick height, from drop-calibrate pairs.

    Each pair is where the robot released the block (its TCP pose with the
    block resting on the table) and where the camera then measured it.
    """

    def __init__(self, camera_xy: np.ndarray, release_xyz: np.ndarray):
        self.pair_count = len(camera_xy)
        self.error_before_mm = float(
            np.linalg.norm(camera_xy - release_xyz[:, :2], axis=1).mean())
        self._xy, self.xy_error_mm = _fit_by_leave_one_out(camera_xy, release_xyz[:, :2])
        self._z, self.z_error_mm = _fit_by_leave_one_out(
            release_xyz[:, :2], release_xyz[:, 2:3])

    def to_robot(self, camera_xy: tuple[float, float]) -> tuple[float, float]:
        x, y = self._xy(np.asarray(camera_xy, dtype=float))[0]
        return float(x), float(y)

    def pick_z(self, x: float, y: float) -> float:
        """TCP height that held the block resting on the table at (x, y)."""
        return float(self._z(np.array([x, y]))[0, 0])

    def summary(self) -> str:
        return (f"Drop-calibrate correction from {self.pair_count} pairs: camera was "
                f"off by {self.error_before_mm:.1f} mm on average; corrected, "
                f"{self.xy_error_mm:.1f} mm on pairs left out of the fit. Pick "
                f"height predicted to {self.z_error_mm:.1f} mm.")


def load_drop_correction(calibration_path: Path, coords_csv: Path,
                         last: int | None = None) -> DropCorrection:
    """Fit the correction from drop-calibrate rows taken with the current map,
    only the most recent ``last`` pairs if given.

    drop_calibrate.py writes each pair as a ``robot`` row followed by a
    ``camera`` row with the same timestamp, which is how its pairs are told
    apart from rows saved by hand.
    """
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    since = datetime.fromisoformat(data.get("updated_utc", data["created_utc"]))
    rows = []
    if coords_csv.exists():
        with coords_csv.open(newline="", encoding="utf-8") as coords_file:
            rows = [r for r in csv.DictReader(coords_file) if r.get("timestamp")]
    camera, release = [], []
    for robot_row, camera_row in zip(rows, rows[1:]):
        if (robot_row.get("source") == "robot" and camera_row.get("source") == "camera"
                and robot_row["timestamp"] == camera_row["timestamp"]
                and datetime.fromisoformat(robot_row["timestamp"]) >= since):
            camera.append([float(camera_row["x_mm"]), float(camera_row["y_mm"])])
            release.append([float(robot_row[k]) for k in ("x_mm", "y_mm", "z_mm")])
    if last is not None:
        camera, release = camera[-last:], release[-last:]
    if len(camera) < MIN_DROP_PAIRS:
        raise SystemExit(
            f"Found {len(camera)} drop-calibrate pair(s) taken with the current "
            f"camera calibration; need at least {MIN_DROP_PAIRS}. Run "
            "programs/drop_calibrate.py first (and don't run main.py --refine "
            "afterwards: that changes the map and retires the pairs)."
        )
    return DropCorrection(np.asarray(camera), np.asarray(release))
