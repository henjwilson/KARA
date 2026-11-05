"""Gripper orientation planning and the drop-calibrate correction.

Shared by ``programs/drop_calibrate.py`` (which places a block on a grid and
logs where the camera saw it) and ``seek_yellow.py`` (which picks blocks up),
so both reach the table with the same gripper orientations.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6 import RobotClient
from parol6.utils.ik import solve_ik
from pinokin import se3_from_rpy, so3_rpy

from cube_pose import CUBE_SIDE_MM
from drop_pairs import carried_pairs, current_pairs

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
MIN_HEIGHT_PAIRS = 2
# Heading-aware correction: pairs needed, smoothing of the map under it, and how
# near (mm, deg) a drop must be to count, chosen by leaving drops out of the fit.
MIN_HEADING_PAIRS = 12
HEADING_BASE_SMOOTHING = 1.0
HEADING_NEAR_MM = 100.0
HEADING_NEAR_DEG = 20.0
HEADING_SHRINK = 0.3
MIN_HEIGHT_TILT_PAIRS = 8
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
        freedom = f"Ry free over {high - low:.0f} deg" if high > low else "Ry set by the block"
        return (f"Rx={rx:6.1f} Ry={ry:6.1f} Rz={rz:6.1f}  ({freedom}, "
                f"Rx lowered {self.rx_decrease} from 90)")


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


def heading_window(x: float, y: float, table: float, safe_height: float,
                   tilt: int = 0) -> tuple[float, float]:
    """Widest continuous range of headings reaching the table at this tilt: its
    lowest heading and its width in degrees. Width -1 means none reach."""
    top, bottom = table + safe_height, table - BELOW_TABLE_MM
    headings = list(range(-180, 180, HEADING_STEP_DEG))
    ok = [_descent_start(x, y, top, bottom, _orientation(tilt, h)) is not None for h in headings]
    if all(ok):
        return -180.0, 360.0
    if not any(ok):
        return 0.0, -1.0
    gap = ok.index(False)
    run = longest = longest_end = 0
    for i in range(gap + 1, gap + 1 + len(ok)):
        run = run + 1 if ok[i % len(ok)] else 0
        if run > longest:
            longest, longest_end = run, i
    low = headings[0] + (longest_end - longest + 1) * HEADING_STEP_DEG
    return float(low), float((longest - 1) * HEADING_STEP_DEG)


def vertical_yaw_range(x: float, y: float, table: float, safe_height: float) -> float:
    """Widest continuous yaw range (deg) reaching the table with the gripper vertical."""
    return max(0.0, heading_window(x, y, table, safe_height)[1])


def lifts_straight_up(angles: list[float], pose: list[float], top_z: float) -> bool:
    """True if the arm, at joint ``angles`` (deg) and WRF ``pose``, can rise to
    ``top_z`` (mm) without changing orientation."""
    target = np.zeros((4, 4))
    x, y = pose[0] / 1000.0, pose[1] / 1000.0
    q = np.radians(angles)
    for z in np.arange(pose[2] + DESCENT_STEP_MM, top_z + 1e-6, DESCENT_STEP_MM):
        se3_from_rpy(x, y, z / 1000.0, *np.radians(pose[3:]), target)
        result = solve_ik(PAROL6_ROBOT.robot, target, q, quiet_logging=True)
        if not result.success or np.abs(result.q - q).max() > MAX_JOINT_STEP_RAD:
            return False
        q = result.q.copy()
    return True


def plan_heading(x: float, y: float, table: float, safe_height: float, heading: float,
                 tilt: int = 0) -> Plan | None:
    """Orientation at exactly this heading, or None if it doesn't reach the table."""
    heading = (heading + 180.0) % 360.0 - 180.0
    rotation = _orientation(tilt, heading)
    start = _descent_start(x, y, table + safe_height, table - BELOW_TABLE_MM, rotation)
    if start is None:
        return None
    return Plan(np.degrees(start).tolist(), rotation, tilt, (heading, heading))


def plan_pick(x: float, y: float, table: float, safe_height: float, yaw: float,
              max_tilt: int = MAX_RX_DECREASE_DEG, prefer: float = 0.0) -> Plan | None:
    """Orientation that lines the jaws up with a block turned ``yaw`` degrees.

    The jaws close along a tool axis that lies along a world axis at heading 0,
    and either world axis grips a square block, so any heading of ``yaw`` plus
    a multiple of 90 will do. Uses the least tilt that reaches the table, up to
    ``max_tilt``, then the heading nearest ``prefer``. Returns None if no such
    heading can, rather than gripping the block across its corners.
    """
    top, bottom = table + safe_height, table - BELOW_TABLE_MM
    headings = sorted(((yaw + 90 * k + 180) % 360 - 180 for k in range(4)),
                      key=lambda h: abs((h - prefer + 180) % 360 - 180))
    for rx_decrease in range(0, max_tilt + 1, RX_STEP_DEG):
        for heading in headings:
            rotation = _orientation(rx_decrease, heading)
            start = _descent_start(x, y, top, bottom, rotation)
            if start is not None:
                return Plan(np.degrees(start).tolist(), rotation, rx_decrease,
                            (heading, heading))
    return None


def _height_design(xy: np.ndarray, count: int) -> np.ndarray:
    """Terms of the height correction: a tilt over the table once there are
    enough pairs to fit one, otherwise a single value everywhere."""
    if count < MIN_HEIGHT_TILT_PAIRS:
        return np.ones((len(xy), 1))
    return np.column_stack((np.ones(len(xy)), xy))


def _heading(rpy_deg: list[float]) -> float:
    """Heading (the plan's Ry) of a gripper orientation read out as Rx, Ry, Rz."""
    pose = np.zeros((4, 4))
    se3_from_rpy(0.0, 0.0, 0.0, *np.radians(rpy_deg), pose)
    turned = pose[:3, :3] @ TOUCH_ROTATION.T
    return float(np.degrees(np.arctan2(turned[1, 0], turned[0, 0])))


def _local_shift(drop_xy: np.ndarray, headings: np.ndarray, residuals: np.ndarray,
                 xy: tuple[float, float], heading: float) -> np.ndarray:
    """Weighted average of nearby drops' residuals, by distance and heading,
    shrunk towards zero where there are few."""
    distance = np.linalg.norm(drop_xy - xy, axis=1)
    turn = np.abs((headings - heading + 180.0) % 360.0 - 180.0)
    weights = np.exp(-(distance / HEADING_NEAR_MM) ** 2 - (turn / HEADING_NEAR_DEG) ** 2)
    return (weights[:, None] * residuals).sum(axis=0) / (weights.sum() + HEADING_SHRINK)


def _heading_leave_one_out(camera_xy: np.ndarray, drop_xy: np.ndarray,
                           headings: np.ndarray) -> float:
    errors = []
    for i in range(len(camera_xy)):
        keep = np.arange(len(camera_xy)) != i
        fit = _ThinPlate(camera_xy[keep], drop_xy[keep], HEADING_BASE_SMOOTHING)
        residuals = drop_xy[keep] - fit(camera_xy[keep])
        guess = fit(camera_xy[i])[0]
        guess = guess + _local_shift(drop_xy[keep], headings[keep], residuals, guess, headings[i])
        errors.append(float(np.linalg.norm(guess - drop_xy[i])))
    return float(np.mean(errors))


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
    """Camera XY -> robot XY and pick height, from drop-calibrate pairs.

    Each pair is where the robot released the block (its TCP pose with the
    block resting on the table) and where the camera then measured it, so
    camera map error and arm positioning error are both absorbed here.

    The arm's error also changes with the gripper's heading, differently across
    the table. Given release orientations, a smooth map over the table is
    corrected near each pick by the drops nearby made at a similar heading
    (``heading_shift``), so a spot dropped at several headings is learned at
    each of them.

    Pairs taken on top of other cubes measure how the arm's error changes with
    height. That change is smooth, so it is one term linear in height, fitted
    over the table, on top of the table correction, and it interpolates between
    heights. Without such pairs height just adds to Z.
    """

    def __init__(self, camera_xy: np.ndarray, release_xyz: np.ndarray,
                 levels: np.ndarray | None = None, side_mm: float = CUBE_SIDE_MM,
                 release_rpy: list | None = None):
        levels = np.zeros(len(camera_xy), dtype=int) if levels is None else np.asarray(levels)
        on_table = levels == 0
        table_camera, table_release = camera_xy[on_table], release_xyz[on_table]
        self.pair_count = len(table_camera)
        self.height_pair_count = len(camera_xy) - self.pair_count
        self.error_before_mm = float(
            np.linalg.norm(table_camera - table_release[:, :2], axis=1).mean())
        table_rpy = [rpy for rpy, flat in zip(release_rpy or [None] * len(on_table), on_table)
                     if flat]
        self._headings = np.zeros(0)
        if len(table_rpy) >= MIN_HEADING_PAIRS and all(rpy is not None for rpy in table_rpy):
            self._headings = np.array([_heading(rpy) for rpy in table_rpy])
            self._drop_xy = table_release[:, :2]
            self._xy = _ThinPlate(table_camera, self._drop_xy, HEADING_BASE_SMOOTHING)
            self._residuals = self._drop_xy - self._xy(table_camera)
            self.xy_error_mm = _heading_leave_one_out(table_camera, self._drop_xy, self._headings)
        else:
            self._xy, self.xy_error_mm = _fit_by_leave_one_out(table_camera,
                                                               table_release[:, :2])
        self._z, self.z_error_mm = _fit_by_leave_one_out(
            table_release[:, :2], table_release[:, 2:3])
        self._per_mm = np.zeros((3, 3))
        if self.height_pair_count >= MIN_HEIGHT_PAIRS:
            raised = ~on_table
            heights = (levels[raised] * side_mm)[:, None]
            at_table = np.hstack((self._xy(camera_xy[raised]),
                                  self._z(release_xyz[raised, :2]) + heights))
            design = _height_design(camera_xy[raised], self.height_pair_count)
            self._per_mm[:, :design.shape[1]] = np.linalg.lstsq(
                design, (release_xyz[raised] - at_table) / heights, rcond=None)[0].T

    def _height_term(self, xy: np.ndarray, height: float) -> np.ndarray:
        return height * (self._per_mm @ np.array([1.0, *xy]))

    def to_robot(self, camera_xy: tuple[float, float],
                 height: float = 0.0) -> tuple[float, float]:
        xy = self._xy(np.asarray(camera_xy, dtype=float))[0]
        x, y, _ = self._height_term(xy, height) + (*xy, 0.0)
        return float(x), float(y)

    def to_robot_points(self, camera_xy: np.ndarray) -> np.ndarray:
        return self._xy(camera_xy)

    def heading_shift(self, x: float, y: float, heading: float) -> np.ndarray:
        """Correction to ``to_robot``'s (x, y) for a pick at ``heading``: what the
        drops nearby at a similar heading were off by. Zero without headings."""
        if not len(self._headings):
            return np.zeros(2)
        return _local_shift(self._drop_xy, self._headings, self._residuals, (x, y), heading)

    def pick_z(self, x: float, y: float, height: float = 0.0) -> float:
        """TCP height that held a block resting ``height`` above the table at (x, y)."""
        table = float(self._z(np.array([x, y]))[0, 0])
        return table + height + float(self._height_term(np.array([x, y]), height)[2])

    def summary(self) -> str:
        return (f"Drop-calibrate correction from {self.pair_count} pairs: camera was "
                f"off by {self.error_before_mm:.1f} mm on average; corrected, "
                f"{self.xy_error_mm:.1f} mm on pairs left out of the fit. Pick "
                f"height predicted to {self.z_error_mm:.1f} mm; "
                f"{self.height_pair_count} pairs above table height"
                f"{'; heading-aware' if len(self._headings) else ''}.")


def load_drop_correction(calibration_path: Path, coords_csv: Path,
                         last: int | None = None) -> DropCorrection:
    """Fit the correction from the stored pairs, carried into the current map,
    and the pairs taken with it; only the most recent ``last`` if given."""
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera, release, levels, orientations = carried_pairs(data)
    for pair in current_pairs(data, coords_csv):
        camera.append(pair["camera_xy_mm"])
        release.append(pair["release_xyz_mm"])
        levels.append(pair["level"])
        orientations.append(pair["release_rpy_deg"])
    camera, release, levels = np.asarray(camera), np.asarray(release), np.asarray(levels, dtype=int)
    table = np.flatnonzero(levels == 0)
    if last is not None:
        table = table[-last:]
    keep = np.concatenate((table, np.flatnonzero(levels > 0)))
    camera, release, levels = camera[keep], release[keep], levels[keep]
    orientations = [orientations[i] for i in keep]
    if len(table) < MIN_DROP_PAIRS:
        raise SystemExit(
            f"Found {len(table)} drop-calibrate pair(s) usable with the current "
            f"camera calibration; need at least {MIN_DROP_PAIRS}. Run "
            "programs/drop_calibrate.py first."
        )
    return DropCorrection(camera, release, levels, release_rpy=orientations)


def preferred_heading(x: float, y: float, table: float, safe_height: float) -> float:
    """The middle of the headings that reach (x, y) vertically: the one that
    picks and drop calibrate turn the gripper around there."""
    low, width = heading_window(x, y, table, safe_height)
    return 0.0 if width >= 360.0 or width < 0 else low + width / 2.0


def plan_seek_pick(correction: DropCorrection, x: float, y: float, z: float,
                   safe_height: float, yaw: float) -> tuple[float, float, Plan] | None:
    """Plan a pick of the block that ``to_robot`` put at (x, y): the TCP XY and
    the plan, or None if it can't be gripped at its angle from there."""
    plan = plan_pick(x, y, z, safe_height, yaw,
                     prefer=preferred_heading(x, y, z, safe_height))
    if plan is None:
        return None
    heading = plan.ry_range[0]
    dx, dy = correction.heading_shift(x, y, heading)
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return x, y, plan
    shifted = plan_heading(x + dx, y + dy, z, safe_height, heading, plan.rx_decrease)
    return None if shifted is None else (x + dx, y + dy, shifted)
