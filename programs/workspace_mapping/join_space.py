"""
Joint-space move-to-pickup-orientation script for the PAROL6 robot arm.

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script.

Usage:
    Type "x,y,letter" (no spaces) and press Enter to move there, e.g.:
        200,150,T
    Letter selects the pickup orientation:
        T = Top     L = Left     R = Right     B = Bottom     F = Front
    Type "home" to return home.
    Type "quit" to exit.

The arm moves in joint space (move_j) to the given X,Y at the fixed pickup
height (Z=80) with the orientation for that letter, then holds there until
you type the next command.
"""

from parol6 import RobotClient
from parol6.utils.errors import MotionError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 5001
TOOL, TOOL_VARIANT = "PNEUMATIC", "vertical"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

Z = -10  # fixed pickup height (mm)

# letter -> (Rx, Ry, Rz) in degrees
ORIENTATIONS = {
    "T": (0, 90, 180),   # Top
    "B": (0, -25, 270),  # Bottom
    "L": (0, -25, 180),  # Left
    "R": (0, -25, 0),    # Right
    #"F": (0, -25, 90),   # Front
    "F": (0, -69.4, 180)
}

MOVE_SPEED = 0.25


def connect_and_home(rbt: RobotClient) -> None:
    rbt.select_tool(TOOL, TOOL_VARIANT)
    current = rbt.angles()
    if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
        print("Homing...")
        rbt.home(wait=True)
    print("Ready. Type 'x,y,letter' to move (e.g. 200,150,T), 'home' to return home, or 'quit' to exit.")
    print("Letters: T=Top  B=Bottom  L=Left  R=Right  F=Front")


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
                print(f"Actual pose after move: {rbt.pose()}")
                continue

            parts = raw.split(",")
            if len(parts) != 3:
                print("Invalid format. Use x,y,letter (e.g. 200,150,T)")
                continue

            x_str, y_str, letter = parts
            letter = letter.strip().upper()

            try:
                x, y = float(x_str), float(y_str)
            except ValueError:
                print("Invalid format. Use x,y,letter (e.g. 200,150,T)")
                continue

            if letter not in ORIENTATIONS:
                print(f"Invalid letter '{letter}'. Use one of: {', '.join(ORIENTATIONS)}")
                continue

            rx, ry, rz = ORIENTATIONS[letter]
            pose = [x, y, Z, rx, ry, rz]
            print(f"X: '{x}'")
            print(f"Y: '{y}'")
            print(f"Z: '{Z}'")
            print(f"Rx: '{rx}'")
            print(f"Ry: '{ry}'")
            print(f"Rz: '{rz}'")

            rbt.move_j(pose=pose, speed=MOVE_SPEED)
            print(f"Actual pose after move: {rbt.pose()}")


            try:
                rbt.move_j(pose=pose, speed=MOVE_SPEED)
            except MotionError as e:
                print(f"Motion rejected: {e}")
            except Exception as e:
                print(f"Move failed: {e}")
    finally:
        rbt.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()