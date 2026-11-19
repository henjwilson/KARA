"""
Beginning of commander system

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script.

Usage:
    Type "x,y" (no spaces) and press Enter to move there, e.g.:
        200,150
    Type "home" to return home.
    Type "park" to return park
    Type "quit" to exit.

The arm moves in joint space (move_j) to the given pose, then holds there
until you type the next command.
"""
import time
import robot_defs as robot_defs
from parol6 import RobotClient
from parol6.utils.errors import MotionError

HOST, PORT = "127.0.0.1", 5001
TOOL = "PNEUMATIC"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

PARK_ANGLES = [0,250,45,90,0,90]
PARK_TOLERANCE_DEG = 2.0

MOVE_SPEED = 0.25

WORLD_X, WORLD_Y, WORLD_Z = 0,0,50

ORIENTATION = [90,-90,90]
FORWARD = [90,-90,90]
LEFT = [90,40,90]
RIGHT = [90,150,90]
BACKWARD = [0,-15,0]

WORLD_MAP = robot_defs.Box(100,-300,400,300)

MIDDLE_BOX = robot_defs.Box(100,-200,250,200)
FRONT_BOX = robot_defs.Box(200,-300,400,300)
RIGHT_BOX = robot_defs.Box(100,-300,200,-200)
LEFT_BOX = robot_defs.Box(100,200,200,300)

PICK_UP_Z = 58

# -----------------
# Robotic arm connects and then moves to park position
# -----------------
def connect_and_home(rbt: RobotClient) -> None:
    rbt.select_tool(TOOL)
    current = rbt.angles()
    #check if the robot is already close to home
    if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
        print("Homing...")
        rbt.home(wait=True)
    if current is None or max(abs(a - h) for a, h in zip(current, PARK_ANGLES)) > PARK_TOLERANCE_DEG:
        print("Parking...")
        try:
            rbt.move_j(pose=PARK_ANGLES, speed=MOVE_SPEED)
        except MotionError as e:
                print(f"Cannot move to park: {e}")
    print("Ready. Type 'x,y' to move (e.g. 200,150), 'home' to return home, 'park' to go to park, or 'quit' to exit.")


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
            if raw.lower() == "park":
                print("parking...")
                # rbt.home(wait=True)
                rbt.move_j(pose=PARK_ANGLES, speed=MOVE_SPEED)
                continue
            parts = raw.split(",")
            if len(parts) != 2:
                print("Invalid format. Use (x,y) (e.g. 200,150)")
                continue

            try:
                WORLD_X, WORLD_Y = (float(p) for p in parts)
                if not WORLD_MAP.contains(WORLD_X,WORLD_Y):
                    print("ERROR: Out of Bounds")
                    continue
            except ValueError:
                print("Invalid format. Use (x,y) (e.g. 200,150)")
                continue

            ###########################################################################

            if MIDDLE_BOX.contains(WORLD_X,WORLD_Y):
                ORIENTATION = BACKWARD
            elif FRONT_BOX.contains(WORLD_X,WORLD_Y):
                ORIENTATION = FORWARD
            elif LEFT_BOX.contains(WORLD_X,WORLD_Y):
                ORIENTATION = LEFT
            elif RIGHT_BOX.contains(WORLD_X,WORLD_Y):
                ORIENTATION = RIGHT
            else:
                print("Cannot find proper Orientation: ")
                continue

            pose = [WORLD_X,WORLD_Y,WORLD_Z, *ORIENTATION]

            # print(f"X: {x}  Y: {y}  Z: {z}  Rx: {rx}  Ry: {ry}  Rz: {rz}")
            try:
                rbt.move_j(pose=pose, speed=MOVE_SPEED)
                # step down to item
                time.sleep(1)
                rbt.move_l([-PICK_UP_Z, 0, 0, 0, 0, 0], speed=0.25, frame="TRF", rel=True)
                time.sleep(1)
                rbt.move_l([PICK_UP_Z, 0, 0, 0, 0, 0], speed=0.25, frame="TRF", rel=True)
                time.sleep(1)
                # go back to park
                rbt.move_j(pose=PARK_ANGLES, speed=MOVE_SPEED)
                print(f"Actual pose after move: {rbt.pose()}")
            except MotionError as e:
                print(f"Motion rejected: {e}")
            except Exception as e:
                print(f"Move failed: {e}")
    finally:
        rbt.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()