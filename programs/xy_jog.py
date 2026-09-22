"""
XY coordinate-input script for the PAROL6 robot arm.

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script.

Usage:
    Type an "x,y" pair (comma-separated, no spaces) and press Enter to move
    the tool to that XY position, e.g.:
        100,300
    Type "home" to return to the home position.
    Type "quit" to exit.

    Anything outside the configured bounds prints "out of bounds" and the
    robot does not move.
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

# Z and orientation are held fixed for these XY moves. Adjust to match
# whatever height/orientation you actually want the tool at.
Z = 180
RX, RY, RZ = 90, 0, 90

X_BOUNDS = (-300, 300)
# Y_BOUNDS = (15, 400)
Y_BOUNDS = (-130, -10)

MOVE_SPEED = 0.5


def in_bounds(x: float, y: float) -> bool:
    return X_BOUNDS[0] <= x <= X_BOUNDS[1] and Y_BOUNDS[0] <= y <= Y_BOUNDS[1]


def connect_and_home(rbt: RobotClient) -> None:
    rbt.select_tool(TOOL)
    rbt.tool.calibrate()
    current = rbt.angles()
    if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
        print("Homing...")
        rbt.home(wait=True)
    print(f"Ready. X bounds {X_BOUNDS}, Y bounds {Y_BOUNDS}.")
    print("Type 'x,y' to move (e.g. 100,300), 'home' to return home, or 'quit' to exit.")


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
                print(f"Home pose (X, Y, Z, Rx, Ry, Rz): {rbt.pose()}")
                continue

            parts = raw.split(",")
            if len(parts) != 2:
                print("Invalid format. Use x,y (e.g. 100,300)")
                continue

            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                print("Invalid format. Use x,y (e.g. 100,300)")
                continue

            if not in_bounds(x, y):
                print("out of bounds")
                continue
            try:
                # rbt.move_l([x, y, Z, RX, RY, RZ], speed=MOVE_SPEED)
                rbt.move_j([x, y, Z, RX, RY, RZ], speed=MOVE_SPEED)
                print(f"pose (X, Y, Z, Rx, Ry, Rz): {rbt.pose()}")
            except MotionError as e:
                print(f"Motion rejected: {e}")
            except Exception as e:
                print(f"Move failed: {e}")
    finally:
        rbt.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()