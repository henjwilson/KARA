import math
from parol6 import RobotClient

rbt = RobotClient()

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

# Select tool, and home only if not already near the home pose
rbt.select_tool("SSG-48")
rbt.tool.calibrate()
current = rbt.angles()
if (
    current is None
    or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG
):
    rbt.home()