"""
Joint-space move script for the PAROL6 robot arm — full pose input.

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script.

Usage:
    Type "x,y,z,rx,ry,rz" (no spaces) and press Enter to move there, e.g.:
        200,150,80,0,-25,180
    Type "home" to return home.
    Type "quit" to exit.

The arm moves in joint space (move_j) to the given pose, then holds there
until you type the next command.
"""

from parol6 import RobotClient
from parol6.utils.errors import MotionError

HOST, PORT = "127.0.0.1", 5001
TOOL = "PNEUMATIC"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

MOVE_SPEED = 0.25

def connect_and_home(rbt: RobotClient) -> None:
    rbt.select_tool(TOOL)
    # rbt.tool.calibrate()
    current = rbt.angles()
    if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
        print("Homing...")
        rbt.home(wait=True)
        # rbt.set_tcp_offset(0, -11, -6)
    print("Ready. Type 'x,y,z,rx,ry,rz' to move (e.g. 200,150,80,0,-25,180), 'home' to return home, or 'quit' to exit.")


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
            if len(parts) != 6:
                print("Invalid format. Use x,y,z,rx,ry,rz (e.g. 200,150,80,0,-25,180)")
                continue

            try:
                x, y, z, rx, ry, rz = (float(p) for p in parts)
            except ValueError:
                print("Invalid format. Use x,y,z,rx,ry,rz (e.g. 200,150,80,0,-25,180)")
                continue

            pose = [x, y, z, rx, ry, rz]
            print(f"X: {x}  Y: {y}  Z: {z}  Rx: {rx}  Ry: {ry}  Rz: {rz}")

            try:
                rbt.move_j(pose=pose, speed=MOVE_SPEED)
                print(f"Actual pose after move: {rbt.pose()}")
                print(f"Join angles after move: {rbt.angles()}")
            except MotionError as e:
                print(f"Motion rejected: {e}")
            except Exception as e:
                print(f"Move failed: {e}")
    finally:
        rbt.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()