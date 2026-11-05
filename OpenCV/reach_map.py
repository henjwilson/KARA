"""Where the gripper reaches the table vertically with 90 degrees of yaw.

The area is worked out in robot XY, so it belongs to the arm and the table
height, not the camera. Building it takes minutes, so it is saved once:

    .venv/bin/python OpenCV/reach_map.py

and drawn through the current calibration, whose four markers tie the robot
frame to the image. Rebuild it if the table height changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from table_fit import project

REACH_MAP = Path(__file__).with_name("reach_map.npz")
MIN_YAW_DEG = 90
SPACING_MM = 15.0
X_RANGE_MM = (40.0, 460.0)
Y_RANGE_MM = (-450.0, 450.0)
MAX_RADIUS_MM = 560.0
# Points must be at least this far inside the blue box.
BOX_INSET_MM = 15.0
# Must match the height the picks start from, in seek_yellow.py.
SAFE_HEIGHT_MM = 50.0


def load_reach_mask(pixel_to_robot: np.ndarray, frame_size: tuple[int, int],
                    path: Path = REACH_MAP) -> np.ndarray | None:
    """Image mask of the area, or None (and why) if it hasn't been built."""
    if not path.is_file():
        print("No reach map yet; build it with: .venv/bin/python OpenCV/reach_map.py")
        return None
    with np.load(path) as saved:
        widths, x0, y0 = saved["widths"], float(saved["xs"][0]), float(saved["ys"][0])
    grid_to_robot = np.array([[0.0, SPACING_MM, x0], [SPACING_MM, 0.0, y0], [0.0, 0.0, 1.0]])
    grid_to_pixel = np.linalg.inv(np.asarray(pixel_to_robot, dtype=np.float64)) @ grid_to_robot
    area = (widths >= MIN_YAW_DEG).astype(np.uint8) * 255
    warped = cv2.warpPerspective(area, grid_to_pixel, frame_size, flags=cv2.INTER_LINEAR)
    return cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)[1]


def _far_edge(calibration: dict, path: Path) -> np.ndarray:
    """Robot XY carrying the marker border's sides out to the area's far edge."""
    with np.load(path) as saved:
        xs, widths = saved["xs"], saved["widths"]
    far_x = xs[(widths >= MIN_YAW_DEG).any(axis=1)].max() + SPACING_MM
    markers = np.asarray([m["robot_xyz_mm"][:2] for m in calibration["markers"]], dtype=np.float64)
    markers = markers[np.argsort(markers[:, 1])]
    ends = []
    for side in (markers[:2], markers[2:]):
        near, far = side[np.argsort(side[:, 0])]
        ends.append(far + (far - near) * (far_x - far[0]) / (far[0] - near[0]))
    return np.asarray(ends)


def border_extension(calibration: dict, path: Path = REACH_MAP) -> np.ndarray | None:
    """Image points that carry the marker border's sides out to the far edge of
    the area, or None if it hasn't been built."""
    if not path.is_file():
        return None
    to_pixel = np.linalg.inv(np.asarray(calibration["pixel_to_robot_xy"], dtype=np.float64))
    return project(to_pixel, _far_edge(calibration, path))


def test_spots(calibration: dict, path: Path = REACH_MAP) -> tuple[list[tuple[float, float]],
                                                                   list[tuple[float, float]]]:
    """Robot XY inside the blue box, in the area and outside it, each list spread
    out: every spot is as far as possible from those before it, so taking more
    of a list only adds spots. The area's edge is only known to one cell, so
    neither list uses the cells next to it."""
    with np.load(path) as saved:
        xs, ys, widths = saved["xs"], saved["ys"], saved["widths"]
    area = (widths >= MIN_YAW_DEG).astype(np.uint8)
    edge = np.ones((3, 3), np.uint8)
    inside, outside = cv2.erode(area, edge) > 0, cv2.dilate(area, edge) == 0
    markers = np.asarray([m["robot_xyz_mm"][:2] for m in calibration["markers"]], dtype=np.float32)
    box = cv2.convexHull(np.vstack((markers, _far_edge(calibration, path).astype(np.float32))))
    in_box = np.array([[cv2.pointPolygonTest(box, (float(x), float(y)), True) >= BOX_INSET_MM
                        for y in ys] for x in xs])
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    return _spread(grid[inside & in_box]), _spread(grid[outside & in_box])


def _spread(candidates: np.ndarray) -> list[tuple[float, float]]:
    if not len(candidates):
        return []
    order = [int(np.argmin(np.linalg.norm(candidates - candidates.mean(axis=0), axis=1)))]
    nearest = np.linalg.norm(candidates - candidates[order[0]], axis=1)
    while len(order) < len(candidates):
        order.append(int(np.argmax(nearest)))
        nearest = np.minimum(nearest, np.linalg.norm(candidates - candidates[order[-1]], axis=1))
    return [(float(x), float(y)) for x, y in candidates[order]]


def work_border(border: np.ndarray, extension: np.ndarray | None) -> np.ndarray:
    """The marker border, reaching out to the far edge of the area if given."""
    if extension is None:
        return border
    points = np.concatenate((np.reshape(border, (-1, 2)), extension))
    return cv2.convexHull(points.astype(np.float32))


def build(calibration_path: Path, path: Path = REACH_MAP) -> None:
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    from placement import TOOL, TOOL_VARIANT, table_z, vertical_yaw_range

    PAROL6_ROBOT.apply_tool(TOOL, TOOL_VARIANT)
    z_plane = np.asarray(json.loads(calibration_path.read_text(encoding="utf-8"))["table_z_plane"])
    xs = np.arange(X_RANGE_MM[0], X_RANGE_MM[1] + 1, SPACING_MM)
    ys = np.arange(Y_RANGE_MM[0], Y_RANGE_MM[1] + 1, SPACING_MM)
    widths = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if np.hypot(x, y) <= MAX_RADIUS_MM:
                table = table_z(z_plane, float(x), float(y))
                widths[i, j] = vertical_yaw_range(float(x), float(y), table, SAFE_HEIGHT_MM)
        print(f"X={x:.0f}: {int((widths[i] >= MIN_YAW_DEG).sum())} of {len(ys)} points", flush=True)
    np.savez_compressed(path, xs=xs, ys=ys, widths=widths)
    area_cm2 = (widths >= MIN_YAW_DEG).sum() * SPACING_MM ** 2 / 100
    print(f"Saved {path}: {area_cm2:.0f} cm2 with {MIN_YAW_DEG} deg of vertical yaw.")


if __name__ == "__main__":
    from main import DEFAULT_CALIBRATION

    build(DEFAULT_CALIBRATION)
