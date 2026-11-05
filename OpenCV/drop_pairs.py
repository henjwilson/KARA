"""Keep drop-calibrate pairs usable when the camera map changes.

A pair's release pose is a robot position, so it never goes stale. Its camera
position is in the map of the calibration it was taken with. Each pair set is
stored with where that map put the four marker spots, untouched, so it can be
carried into any later map: the spots are the same physical points, and fitting
one set of them onto the other gives the change between the two maps. Nothing
is rewritten, so a bad calibration can't damage the pairs.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from table_fit import fit_similarity, project

# A camera move fits the spots to about 1 mm; a marker pushed out of place
# fits to tens of mm. Above this limit the map is not the same table layout.
MAX_SPOT_MISFIT_MM = 5.0


def _spot(robot_xy: list[float]) -> tuple[float, float]:
    return (round(robot_xy[0], 2), round(robot_xy[1], 2))


def marker_spots(data: dict) -> dict[tuple[float, float], list[float]]:
    """Where this calibration's map puts each marker's touched spot.

    Keyed by the spot's robot XY rather than the marker ID, because a
    calibration can reassign which marker sits on which spot.
    """
    pixels = np.asarray([m["image_xy"] for m in data["markers"]], dtype=np.float64)
    mapped = project(np.asarray(data["pixel_to_robot_xy"], dtype=np.float64), pixels)
    return {_spot(m["robot_xyz_mm"]): xy.tolist()
            for m, xy in zip(data["markers"], mapped)}


def current_pairs(data: dict, coords_csv: Path) -> list[dict]:
    """Pairs in the CSV taken with this calibration's map.

    drop_calibrate.py writes each pair as a ``robot`` row followed by a
    ``camera`` row with the same timestamp, which is how its pairs are told
    apart from rows saved by hand.
    """
    since = datetime.fromisoformat(data.get("updated_utc", data["created_utc"]))
    rows = []
    if coords_csv.exists():
        with coords_csv.open(newline="", encoding="utf-8") as coords_file:
            rows = [r for r in csv.DictReader(coords_file) if r.get("timestamp")]
    return [
        {"camera_xy_mm": [float(camera_row["x_mm"]), float(camera_row["y_mm"])],
         "release_xyz_mm": [float(robot_row[k]) for k in ("x_mm", "y_mm", "z_mm")],
         "release_rpy_deg": [float(robot_row[k]) for k in ("rx_deg", "ry_deg", "rz_deg")],
         "level": int(robot_row.get("level") or 0)}
        for robot_row, camera_row in zip(rows, rows[1:])
        if robot_row.get("source") == "robot" and camera_row.get("source") == "camera"
        and robot_row["timestamp"] == camera_row["timestamp"]
        and datetime.fromisoformat(robot_row["timestamp"]) >= since
    ]


def archive_pairs(data: dict, coords_csv: Path) -> list[dict]:
    """Stored pair sets plus this map's pairs; call before the map changes."""
    drop_sets = list(data.get("drop_sets", []))
    pairs = current_pairs(data, coords_csv)
    if pairs:
        spots = [{"spot_mm": list(spot), "map_xy_mm": xy}
                 for spot, xy in marker_spots(data).items()]
        drop_sets.append({"spots": spots, "pairs": pairs})
    return drop_sets


def carried_pairs(data: dict) -> tuple[list[list[float]], list[list[float]], list[int],
                                       list[list[float] | None]]:
    """Stored pairs in this calibration's map: camera XY, release XYZ, level and
    release orientation (None for pairs stored without it).

    A set whose spots don't fit is left out, and stays stored: it applies
    again once the markers are back.
    """
    now = marker_spots(data)
    camera, release, levels, orientations = [], [], [], []
    for drop_set in data.get("drop_sets", []):
        then = {_spot(s["spot_mm"]): s["map_xy_mm"] for s in drop_set["spots"]}
        if set(then) != set(now):
            print(f"Skipping {len(drop_set['pairs'])} stored drop pairs: they were "
                  "taken with different marker coordinates.")
            continue
        source = np.asarray([then[spot] for spot in now], dtype=np.float64)
        target = np.asarray(list(now.values()), dtype=np.float64)
        change = fit_similarity(source, target)
        misfit = float(np.linalg.norm(project(change, source) - target, axis=1).max())
        if misfit > MAX_SPOT_MISFIT_MM:
            print(f"Skipping {len(drop_set['pairs'])} stored drop pairs: the markers "
                  f"are up to {misfit:.0f} mm from where they were, not just the "
                  "camera. Put them back and recalibrate to use these pairs again.")
            continue
        points = np.asarray([p["camera_xy_mm"] for p in drop_set["pairs"]], dtype=np.float64)
        camera += project(change, points).tolist()
        release += [p["release_xyz_mm"] for p in drop_set["pairs"]]
        levels += [p.get("level", 0) for p in drop_set["pairs"]]
        orientations += [p.get("release_rpy_deg") for p in drop_set["pairs"]]
    return camera, release, levels, orientations
