"""Fit the camera-pixel -> robot-XY table mapping from ArUco markers.

A plain 4-point homography reproduces whatever error is in the touched robot
coordinates as warping: stretched axes, skew, even a mirrored map. Here the
shape of the mapping comes from the camera instead: every marker is a printed
square of the same size, so its 4 detected corners must map to an equal-sized
square on the table. The touched marker centers only place, rotate, and scale
that shape, and with 4 of them their errors average out.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

# ArUco corner order (TL, TR, BR, BL) seen from above with robot Y up.
UNIT_SQUARE = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
# Expected errors: corners are sub-pixel (~0.3 mm on the table); touching a
# marker center by hand is good to a few mm.
CORNER_SIGMA_MM = 0.3
TOUCH_SIGMA_MM = 5.0
# A different ID pairing must fit this many times better than the configured
# one before it is treated as a mix-up.
PAIRING_MARGIN = 4.0


@dataclass
class TableFit:
    homography: np.ndarray  # pixel -> robot XY mm
    pairing: tuple[int, ...]  # pairing[i] = index of the robot point for marker i
    centers_mm: np.ndarray  # fitted marker centers
    marker_size_mm: float
    touch_residuals_mm: np.ndarray  # fitted center - touched point, per marker
    corner_rms_mm: float  # how far detected corners are from perfect squares
    cost: float


def _normalizer(pts: np.ndarray) -> np.ndarray:
    center = pts.mean(axis=0)
    scale = np.sqrt(2) / np.linalg.norm(pts - center, axis=1).mean()
    return np.array([[scale, 0, -scale * center[0]],
                     [0, scale, -scale * center[1]],
                     [0, 0, 1.0]])


def project(homography: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64) @ homography[:, :2].T + homography[:, 2]
    return p[:, :2] / p[:, 2:3]


def _levenberg_marquardt(residuals, x: np.ndarray, iterations: int = 200):
    damping = 1e-3
    r = residuals(x)
    cost = float(r @ r)
    for _ in range(iterations):
        jacobian = np.empty((r.size, x.size))
        for i in range(x.size):
            dx = np.zeros_like(x)
            dx[i] = 1e-7 * max(1.0, abs(x[i]))
            jacobian[:, i] = (residuals(x + dx) - r) / dx[i]
        jtj, grad = jacobian.T @ jacobian, jacobian.T @ r
        while True:
            step = np.linalg.solve(jtj + damping * np.diag(np.diag(jtj) + 1e-12), -grad)
            r_new = residuals(x + step)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                improvement = cost - cost_new
                x, r, cost = x + step, r_new, cost_new
                damping = max(damping / 3, 1e-9)
                break
            damping *= 4
            if damping > 1e9:
                return x, cost
        if improvement < 1e-12 * max(cost, 1e-12):
            break
    return x, cost


def _fit_pairing(corners: np.ndarray, touched: np.ndarray,
                 marker_size_mm: float | None) -> TableFit:
    count = len(corners)
    flat = corners.reshape(-1, 2)
    # Work in normalized coordinates so every homography entry is O(1).
    norm_img, norm_world = _normalizer(flat), _normalizer(touched)
    world_from_norm = np.linalg.inv(norm_world)
    img = project(norm_img, flat)

    h0, _ = cv2.findHomography(corners.mean(axis=1), touched)
    if h0 is None:
        h0 = cv2.getPerspectiveTransform(
            corners.mean(axis=1).astype(np.float32), touched.astype(np.float32)
        )
    hn0 = norm_world @ h0 @ np.linalg.inv(norm_img)
    hn0 /= hn0[2, 2]
    mapped0 = project(h0, flat).reshape(count, 4, 2)
    side0 = float(np.linalg.norm(mapped0 - np.roll(mapped0, 1, axis=1), axis=2).mean())
    angles0 = [np.arctan2(m[1, 1] - m[0, 1], m[1, 0] - m[0, 0]) for m in mapped0]
    x0 = np.concatenate([hn0.ravel()[:8], touched.ravel(), angles0,
                         [] if marker_size_mm else [side0]])

    def unpack(x):
        hn = np.append(x[:8], 1.0).reshape(3, 3)
        centers = x[8:8 + 2 * count].reshape(count, 2)
        angles = x[8 + 2 * count:8 + 3 * count]
        side = marker_size_mm or x[-1]
        return hn, centers, angles, side

    def squares(centers, angles, side):
        cos, sin = np.cos(angles), np.sin(angles)
        rot = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)
        return centers[:, None, :] + side * np.einsum("kij,nj->kni", rot, UNIT_SQUARE)

    def residuals(x):
        hn, centers, angles, side = unpack(x)
        mapped = project(world_from_norm, project(hn, img)).reshape(count, 4, 2)
        return np.concatenate([
            ((mapped - squares(centers, angles, side)) / CORNER_SIGMA_MM).ravel(),
            ((centers - touched) / TOUCH_SIGMA_MM).ravel(),
        ])

    x, cost = _levenberg_marquardt(residuals, x0)
    hn, centers, angles, side = unpack(x)
    homography = world_from_norm @ hn @ norm_img
    homography /= homography[2, 2]
    mapped = project(homography, flat).reshape(count, 4, 2)
    corner_err = mapped - squares(centers, angles, side)
    return TableFit(
        homography=homography,
        pairing=tuple(range(count)),
        centers_mm=centers,
        marker_size_mm=float(side),
        touch_residuals_mm=np.linalg.norm(centers - touched, axis=1),
        corner_rms_mm=float(np.sqrt((corner_err ** 2).sum(axis=2).mean())),
        cost=cost,
    )


def fit_table(corners: np.ndarray, touched: np.ndarray,
              marker_size_mm: float | None = None) -> tuple[TableFit, TableFit]:
    """Fit the mapping, checking every way of pairing markers to robot points.

    ``corners`` is (k, 4, 2) image pixels in ArUco order; ``touched`` is (k, 2)
    robot XY of each marker's center, in the same marker order. Returns the
    fit for the given pairing and the best-fitting pairing (often the same).
    """
    corners = np.asarray(corners, dtype=np.float64)
    touched = np.asarray(touched, dtype=np.float64)
    given = _fit_pairing(corners, touched, marker_size_mm)
    best = given
    for pairing in itertools.permutations(range(len(touched))):
        if pairing == given.pairing:
            continue
        fit = _fit_pairing(corners, touched[list(pairing)], marker_size_mm)
        if fit.cost < best.cost:
            fit.pairing = pairing
            best = fit
    return given, best


def is_mirrored(homography: np.ndarray, at_px: np.ndarray) -> bool:
    """True if the map flips the table over (a camera can't see it mirrored).

    Image Y points down and robot Y up, so an unmirrored view has a negative
    Jacobian determinant.
    """
    eps = 1.0
    p = project(homography, np.array([at_px, at_px + (eps, 0), at_px + (0, eps)]))
    jac = np.column_stack([(p[1] - p[0]) / eps, (p[2] - p[0]) / eps])
    return float(np.linalg.det(jac)) > 0


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation + uniform scale + translation (3x3) mapping
    ``src`` onto ``dst``; reflections are excluded."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_src, dst - mu_dst
    u, sigma, vt = np.linalg.svd(b.T @ a)
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt)) or 1.0])
    rotation = u @ d @ vt
    scale = float((sigma * np.diag(d)).sum() / (a ** 2).sum())
    out = np.eye(3)
    out[:2, :2] = scale * rotation
    out[:2, 2] = mu_dst - scale * rotation @ mu_src
    return out
