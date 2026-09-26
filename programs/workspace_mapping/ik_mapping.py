"""
Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script.
"""

from parol6 import RobotClient
from parol6.utils.errors import MotionError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 5001
TOOL = "SSG-48"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
X_RANGE = range(100, 351, 10)   # 100 to 350mm, step 10
Y_RANGE = range(-260, 261, 10)  # -260 to 260mm, step 10
Z = 80


ORIENTATIONS = {
    "TOP":   (rx1, ry1, rz1),
    # rz fixed: -90
    # 
    "FRONT": (rx2, ry2, rz2),
    "LEFT":  (rx3, ry3, rz3),
    "RIGHT": (rx4, ry4, rz4),
    "BACK":  (rx5, ry5, rz5),
}

results = {}

for x in X_RANGE:
    for y in Y_RANGE:
        point_result = {}
        for name, (rx, ry, rz) in ORIENTATIONS.items():
            valid = check_orientation(x, y, Z, rx, ry, rz)  # <- real IK call goes here
            point_result[name] = valid
        results[(x, y)] = point_result