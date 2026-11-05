"""Fit a real 3D cube to each block's outline.

Against the black table a cube's outline is the most reliable thing in the
image. It doesn't depend on how the faces are lit, and it is the hull of the
cube's eight projected corners, so every visible edge constrains one pose
instead of each edge being fitted on its own. Given the camera, that pose gives
the block's true table position and yaw at once, whatever its height.

The camera comes from the table map: its focal length is the only thing a map
can't supply, and it doesn't change when the camera moves, so T T stays enough.
FOCAL_PX and CUBE_SIDE_MM were measured by fitting 24 real cubes at 1280x720,
which gave 0.54 px of outline error; lens distortion there is about 1 px at the
image edges, so it isn't modeled.
"""

from __future__ import annotations

import cv2
import numpy as np

FOCAL_PX = 925.0
CUBE_SIDE_MM = 15.0
# Cube if this much brighter than the frame's median, which is the black table.
BLACK_MARGIN = 8
MAX_LEVELS = 4
# Outline error (px) of a good fit, and the limit past which it isn't a cube.
GOOD_FIT_PX = 0.6
MAX_FIT_PX = 1.5
# An outline only counts as one cube or one stack if it fits this well.
STACK_FIT_PX = 1.0
# A stack must fit this much better than one cube, and outline more than one
# cube's area: near the point under the camera height barely shows, and a false
# stack would send the arm a whole cube too high.
STACK_GAIN = 0.6
MIN_STACK_AREA = 1.2
# Outline points further off than this (px) belong to something else, such as
# a touching neighbour.
OUTLIER_PX = 3.0
MAX_OUTLINE_POINTS = 40
# How far (px) an outline point may move to reach the true edge.
EDGE_SEARCH_PX = 2.5
# Outline area, in single cubes, of two cubes side by side, and past which a
# blob isn't only cubes.
PAIR_AREA_RANGE = (1.5, 2.6)
MAX_BLOB_AREA = 7.0
UNIT_SQUARE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


class CubeCamera:
    """The camera, from a table map, with table height as Z."""

    def __init__(self, pixel_to_robot: np.ndarray, frame_size: tuple[int, int],
                 side_mm: float = CUBE_SIDE_MM, x_range: tuple[float, float] = (120.0, 320.0),
                 y_range: tuple[float, float] = (-280.0, 300.0)):
        width, height = frame_size
        focal = FOCAL_PX * width / 1280.0
        self.K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
        self.side = side_mm
        to_pixel = np.linalg.inv(pixel_to_robot)
        self.to_pixel = to_pixel
        # The table map isn't quite a rigid view; these are the camera poses
        # closest to it over the area it was fitted on.
        xs, ys = np.meshgrid(np.linspace(*x_range, 5), np.linspace(*y_range, 5))
        table = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
        pixels = cv2.perspectiveTransform(table[:, :2].reshape(-1, 1, 2), to_pixel).reshape(-1, 2)
        _, rotation, translation = cv2.solvePnP(table, pixels, self.K, None,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        self.R = cv2.Rodrigues(rotation)[0]
        self.t = translation.ravel()
        self.position = -self.R.T @ self.t

    def project(self, xyz: np.ndarray) -> np.ndarray:
        camera = xyz @ self.R.T + self.t
        pixels = camera @ self.K.T
        return pixels[..., :2] / pixels[..., 2:3]

    def cube_area(self, pixel: np.ndarray) -> float:
        """Image area of one cube standing where ``pixel`` shows it."""
        pose = np.array([*self.on_plane(pixel, self.side / 2.0), 0.0])
        corners = self.project(_column(pose, self.side, self.side)).astype(np.float32)
        return float(cv2.contourArea(cv2.convexHull(corners)))

    def table_pixels(self, xy: np.ndarray) -> np.ndarray:
        """Where the table map itself puts robot XY, so those pixels map back exactly."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.to_pixel).reshape(-1, 2)

    def on_plane(self, pixel: np.ndarray, height: float) -> np.ndarray:
        """Robot XY of the point seen at ``pixel``, at ``height`` above the table."""
        ray = self.R.T @ np.linalg.solve(self.K, np.array([pixel[0], pixel[1], 1.0]))
        return (self.position + ray * (height - self.position[2]) / ray[2])[:2]


def _footprint(pose: np.ndarray, side: float) -> np.ndarray:
    cos, sin = np.cos(pose[2]), np.sin(pose[2])
    return (UNIT_SQUARE * side) @ np.array([[cos, sin], [-sin, cos]]) + pose[:2]


def _column(pose: np.ndarray, side: float, top: float) -> np.ndarray:
    """Corners of a column ``top`` high standing on the table."""
    square = _footprint(pose, side)
    return np.vstack((np.column_stack((square, np.zeros(4))),
                      np.column_stack((square, np.full(4, top)))))


def _distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest polygon edge."""
    start, edge = polygon, np.roll(polygon, -1, axis=0) - polygon
    offset = points[:, None] - start
    along = np.clip((offset * edge).sum(2) / np.maximum((edge * edge).sum(1), 1e-9), 0.0, 1.0)
    return np.linalg.norm(offset - along[..., None] * edge, axis=2).min(axis=1)


def _errors(pose: np.ndarray, points: np.ndarray, camera: CubeCamera, top: float) -> np.ndarray:
    corners = camera.project(_column(pose, camera.side, top))
    hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    return _distances(points, hull)


def _cost(errors: np.ndarray) -> float:
    return float((np.minimum(errors, OUTLIER_PX) ** 2).sum())


def _on_edges(outline: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Outline points moved along their normals onto the true edge, to a
    fraction of a pixel.

    Thresholding puts an edge in a different place next to a bright face than a
    dim one, which tilts a fit. The steepest change doesn't depend on the face.
    """
    points = outline.reshape(-1, 2).astype(np.float32)
    points = points[:: max(1, len(points) // MAX_OUTLINE_POINTS)]
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    offsets = np.arange(-EDGE_SEARCH_PX, EDGE_SEARCH_PX + 0.01, 0.25, dtype=np.float32)
    samples = points[:, None] + offsets[None, :, None] * normal[:, None]
    profile = cv2.remap(value, samples[..., 0], samples[..., 1], cv2.INTER_LINEAR)
    step = np.abs(np.diff(profile, axis=1))
    peak = np.clip(step.argmax(axis=1), 1, step.shape[1] - 2)
    rows = np.arange(len(points))
    left, middle, right = step[rows, peak - 1], step[rows, peak], step[rows, peak + 1]
    curve = left - 2 * middle + right
    shift = 0.5 * (left - right) / np.where(np.abs(curve) < 1e-6, -1.0, curve)
    return points + ((peak + 0.5 + shift) * 0.25 - EDGE_SEARCH_PX)[:, None] * normal


def fit_cube(outline: np.ndarray, camera: CubeCamera,
             top: float) -> tuple[np.ndarray, float]:
    """Pose (x, y, yaw in radians) of a column ``top`` high, and its outline
    error in pixels."""
    points = outline.reshape(-1, 2).astype(np.float64)
    xy = camera.on_plane(points.mean(axis=0), top / 2.0)
    starts = [np.array([xy[0], xy[1], angle]) for angle in np.radians(np.arange(0.0, 90.0, 18.0))]
    pose = min(starts, key=lambda p: _cost(_errors(p, points, camera, top)))
    steps = np.diag([0.2, 0.2, 0.02])
    for _ in range(10):
        errors = _errors(pose, points, camera, top)
        jacobian = np.stack([(_errors(pose + step, points, camera, top) - errors) / np.linalg.norm(step)
                             for step in steps], axis=1)
        weights = 1.0 / np.maximum(errors, 1.0)
        weighted = jacobian * weights[:, None]
        update = np.linalg.solve(weighted.T @ jacobian + 1e-3 * np.eye(3), -weighted.T @ errors)
        if _cost(_errors(pose + update, points, camera, top)) >= _cost(errors):
            break
        pose = pose + update
        if np.abs(update[:2]).max() < 0.02 and abs(update[2]) < 0.001:
            break
    errors = np.minimum(_errors(pose, points, camera, top), OUTLIER_PX)
    return pose, float(np.sqrt((errors ** 2).mean()))


def _apply_pose(block: dict, pose: np.ndarray, error: float, level: int,
                camera: CubeCamera) -> None:
    top = (level + 1) * camera.side
    square = _footprint(pose, camera.side)
    block["level"] = level
    block["height_mm"] = level * camera.side
    block["fit_error_px"] = error
    block["confidence"] = float(np.clip((MAX_FIT_PX - error) / (MAX_FIT_PX - GOOD_FIT_PX), 0.0, 1.0))
    block["top_found"] = error <= MAX_FIT_PX
    block["top_box"] = camera.project(np.column_stack((square, np.full(4, top)))).astype(np.float32)
    block["top_xy"] = tuple(float(c) for c in camera.project(np.array([*pose[:2], top])))
    block["base_box"] = camera.table_pixels(square)
    block["contact_xy"] = tuple(float(c) for c in camera.table_pixels(pose[:2])[0])


def _fit_alone(block: dict, outline: np.ndarray, value: np.ndarray, camera: CubeCamera) -> None:
    _apply_pose(block, *fit_cube(_on_edges(outline, value), camera, camera.side), 0, camera)


def _outline(mask: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return max(found, key=cv2.contourArea).reshape(-1, 2) + origin


def _separate(blocks: list[dict], mask: np.ndarray, frame: np.ndarray,
              origin: tuple[int, int]) -> list[np.ndarray]:
    """Each touching block's share of a blob, dark sides included, seeded by
    its color so the boundary lands on the real edge between them."""
    x, y = origin
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[mask == 0] = 1
    for index, block in enumerate(blocks):
        cv2.drawContours(markers, [(block["contour"] - origin).reshape(-1, 1, 2)], -1, index + 2, -1)
    height, width = mask.shape
    cv2.watershed(frame[y:y + height, x:x + width], markers)
    return [_outline((markers == index + 2).astype(np.uint8), origin)
            for index in range(len(blocks))]


def _halves(mask: np.ndarray, origin: tuple[int, int]) -> list[np.ndarray]:
    """A blob cut across its length: two cubes flush side by side have no neck."""
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    axis = np.linalg.eigh(np.cov((points - center).T))[1][:, -1]
    grid_x, grid_y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    side = (grid_x - center[0]) * axis[0] + (grid_y - center[1]) * axis[1] > 0
    return [_outline((mask * part).astype(np.uint8), origin) for part in (side, ~side)]


def refine_blocks(blocks: list[dict], frame: np.ndarray, value: np.ndarray,
                  camera: CubeCamera) -> list[dict]:
    """Replace each block's estimates with its fitted cube.

    The outline comes from everything brighter than the table, so dark side
    faces count. One outline can hold a stack, which fits as a taller column,
    or touching cubes, which are separated first. ``top_found`` then means a
    good fit. Blocks under a stack are dropped.
    """
    black = float(np.median(value)) + BLACK_MARGIN
    bright = (value > black).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    groups: dict[int, list[dict]] = {}
    for block in blocks:
        u, v = block["contour"][:, 0], block["contour"][:, 1]
        groups.setdefault(int(np.bincount(labels[v, u]).argmax()), []).append(block)
    value = value.astype(np.float32)
    kept = []
    for label, group in groups.items():
        left, top, width, height, area = stats[label]
        cubes = area / camera.cube_area(np.asarray(group[0]["center_xy"]))
        if label == 0 or cubes > MAX_BLOB_AREA:
            for block in group:
                _fit_alone(block, block["contour"], value, camera)
            kept += group
            continue
        mask = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
        origin = (int(left), int(top))
        outline = _outline(mask, origin)
        edges = _on_edges(outline, value)
        column = None
        pose, single_error = fit_cube(edges, camera, camera.side)
        if single_error <= STACK_FIT_PX:
            column = pose, single_error, 0
        elif cubes >= MIN_STACK_AREA:
            for level in range(1, MAX_LEVELS):
                pose, error = fit_cube(edges, camera, (level + 1) * camera.side)
                if error <= STACK_FIT_PX and error <= STACK_GAIN * single_error:
                    column = pose, error, level
                    break
        if column is not None:
            # Each color in a stack is seen in the same outline; the block
            # that counts is the one at the column's top.
            pose, error, level = column
            seen_at = camera.project(np.array([*pose[:2], (level + 0.5) * camera.side]))
            block = min(group, key=lambda b: np.linalg.norm(np.subtract(b["center_xy"], seen_at)))
            _apply_pose(block, pose, error, level, camera)
            kept.append(block)
        elif len(group) > 1:
            for block, part in zip(group, _separate(group, mask, frame, origin)):
                _fit_alone(block, part, value, camera)
            kept += group
        elif PAIR_AREA_RANGE[0] <= cubes <= PAIR_AREA_RANGE[1]:
            for part in _halves(mask, origin):
                half = dict(group[0], contour=part, center_xy=tuple(part.mean(axis=0).tolist()))
                _fit_alone(half, part, value, camera)
                kept.append(half)
        else:
            _fit_alone(group[0], group[0]["contour"], value, camera)
            kept += group
    return kept
