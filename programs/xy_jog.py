"""
XYZ coordinate-input script for the PAROL6 robot arm.

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script.

Usage:
    Type a target separated by spaces or commas and press Enter:
        x y z                 move the tool to that point, mm (WRF)
        x y z rx ry rz        ...with that orientation, degrees
        x y                   keep the current Z
    Without an orientation, the one used when touching the table markers on
    that side is tried first: (90, 0, 90) for Y >= 0, (-90, 0, -90) for Y < 0.
    If that can't reach the point (e.g. mid-table), the same gripper attitude
    is turned about vertical to face the robot base, as seek_yellow.py does.

    "pose" prints the current pose, "home" returns home, "quit" exits.
    Anything outside the configured bounds prints "out of bounds" and the
    robot does not move.
"""

import math

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6 import RobotClient
from parol6.utils.errors import MotionError
from parol6.utils.ik import solve_ik

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 5001
TOOL, TOOL_VARIANT = "PNEUMATIC", "vertical"

ORIENTATION_POS_Y = (90.0, 0.0, 90.0)
ORIENTATION_NEG_Y = (-90.0, 0.0, -90.0)

X_BOUNDS = (-300, 500)
Y_BOUNDS = (-400, 400)
# The table surface is around Z -20 to +10 mm at the gripper tip.
Z_BOUNDS = (15, 450)

MOVE_SPEED = 0.3

# Gripper attitude of ORIENTATION_POS_Y as a rotation matrix; the fallback
# rotates it about vertical only.
_TOUCH_ROTATION = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
HEADING_OFFSETS_DEG = sorted(range(-180, 180, 10), key=abs)


def in_bounds(x: float, y: float, z: float) -> bool:
    return (
        X_BOUNDS[0] <= x <= X_BOUNDS[1]
        and Y_BOUNDS[0] <= y <= Y_BOUNDS[1]
        and Z_BOUNDS[0] <= z <= Z_BOUNDS[1]
    )


def print_pose(rbt: RobotClient) -> None:
    pose = rbt.pose()
    if pose is None:
        print("Pose unavailable.")
        return
    print(
        f"Position WRF (mm):  X={pose[0]:8.2f}   Y={pose[1]:8.2f}   Z={pose[2]:8.2f}\n"
        f"Rotation WRF (°):  Rx={pose[3]:8.2f}  Ry={pose[4]:8.2f}  Rz={pose[5]:8.2f}"
    )


def connect_and_home(rbt: RobotClient) -> None:
    rbt.select_tool(TOOL, TOOL_VARIANT)
    PAROL6_ROBOT.apply_tool(TOOL, TOOL_VARIANT)
    # A robot fresh from power-on sits at the home angles but unreferenced, so
    # always home; once referenced this is just a short planned move.
    print("Homing...")
    rbt.home(wait=True)
    print(f"Ready. X bounds {X_BOUNDS}, Y bounds {Y_BOUNDS}, Z bounds {Z_BOUNDS}.")
    print("Type 'x y' (current Z), 'x y z', or 'x y z rx ry rz' to move; 'pose', 'home', or 'quit'.")


def parse_target(raw: str, rbt: RobotClient) -> tuple[list[float], bool] | None:
    """Return ([x, y, z, rx, ry, rz], orientation_given), or None after
    printing why the input is invalid."""
    try:
        parts = raw.split(",") if "," in raw else raw.split()
        values = [float(part.strip()) for part in parts]
    except ValueError:
        values = []
    if len(values) not in (2, 3, 6):
        print("Invalid format. Use x y, x y z, or x y z rx ry rz (e.g. 250 100 120)")
        return None
    if len(values) == 2:
        current = rbt.pose()
        if current is None:
            print("Pose unavailable; type a Z too.")
            return None
        values.append(current[2])
    given = len(values) == 6
    if not given:
        values.extend(ORIENTATION_POS_Y if values[1] >= 0 else ORIENTATION_NEG_Y)
    return values, given


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def solve_facing_base(rbt: RobotClient, x: float, y: float, z: float) -> list[float] | None:
    """Joint angles (deg) reaching (x, y, z) with the gripper facing the base."""
    current = rbt.angles()
    if current is None:
        return None
    seed = np.radians(current)
    # _TOUCH_ROTATION's tool-Z column is -Y, i.e. heading -90 deg.
    base_heading = math.degrees(math.atan2(-y, -x)) + 90.0
    target = np.eye(4)
    target[:3, 3] = (x / 1000.0, y / 1000.0, z / 1000.0)
    for offset in HEADING_OFFSETS_DEG:
        target[:3, :3] = _rot_z(base_heading + offset) @ _TOUCH_ROTATION
        result = solve_ik(PAROL6_ROBOT.robot, target, seed, quiet_logging=True)
        if result.success:
            return np.degrees(result.q).tolist()
    return None


# move to target:
def move_to(rbt: RobotClient, pose: list[float], speed: float = MOVE_SPEED) -> str:
    try:
        rbt.move_l(pose, speed=speed)
        return "straight"
    except MotionError as e:
        print(f"Straight-line path rejected ({e}); retrying with move_j")
        rbt.move_j(pose=pose, speed=speed)
        return "curved"


def main() -> None:
    rbt = RobotClient(host=HOST, port=PORT)
    connect_and_home(rbt)

    try:
        while True:
            raw = input("> ").strip()
            if not raw:
                continue

            if raw.lower() in ("quit", "exit"):
                break

            if raw.lower() == "home":
                print("Homing...")
                rbt.home(wait=True)
                print_pose(rbt)
                continue

            if raw.lower() == "pose":
                print_pose(rbt)
                continue

            parsed = parse_target(raw, rbt)
            if parsed is None:
                continue
            target, orientation_given = parsed
            if not in_bounds(*target[:3]):
                print("out of bounds")
                continue
            try:
                try:
                    path = move_to(rbt, target, MOVE_SPEED)
                except MotionError as e:
                    if orientation_given:
                        raise
                    angles = solve_facing_base(rbt, *target[:3])
                    if angles is None:
                        raise
                    print(f"Not reachable at the side orientation ({e.__class__.__name__}); "
                          "turning the gripper to face the base")
                    rbt.move_j(angles, speed=MOVE_SPEED)
                    path = "curved, gripper facing base"
                print(f"Arrived ({path} path).")
                print_pose(rbt)
            except MotionError as e:
                print(f"Motion rejected (out of reach): {e}")
            except Exception as e:
                print(f"Move failed: {e}")
    finally:
        rbt.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
