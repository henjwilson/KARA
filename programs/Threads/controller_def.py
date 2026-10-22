import threading
import time
from dataclasses import dataclass
from models import Block
from parol6 import RobotClient
from parol6.utils.errors import MotionError

# ---------------------------------------------------------------------------
# Global Variable
# ---------------------------------------------------------------------------
rbt = None
controller_to_opencv = None
opencv_done = None
controller_stop = None
# lock
map_lock =  None
# map
active_blocks = None
# -1 mean replace all. any other means get that id
lookup_index  = None

HOST, PORT = "127.0.0.1", 5001
TOOL = "PNEUMATIC"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

PARK_ANGLES = [0,250,45,90,0,90]
PARK_TOLERANCE_DEG = 2.0

MOVE_SPEED = 0.25

WORLD_X, WORLD_Y, WORLD_Z = 0,0,50
PICK_UP_Z = 58

# gripper orientation
ORIENTATION = [90,-90,90]
FORWARD = [90,-90,90]
LEFT = [90,40,90]
RIGHT = [90,150,90]
BACKWARD = [0,-15,0]

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
@dataclass
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

WORLD_MAP = Box(100,-300,400,300)
MIDDLE_BOX = Box(100,-200,250,200)
FRONT_BOX = Box(200,-300,400,300)
RIGHT_BOX = Box(100,-300,200,-200)
LEFT_BOX = Box(100,200,200,300)

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

''' PICK UP BLOCK
    - move to item and pick up
'''
def pick_up():
    if MIDDLE_BOX.contains(WORLD_X,WORLD_Y):
        ORIENTATION = BACKWARD
    elif FRONT_BOX.contains(WORLD_X,WORLD_Y):
        ORIENTATION = FORWARD
    elif LEFT_BOX.contains(WORLD_X,WORLD_Y):
        ORIENTATION = LEFT
    elif RIGHT_BOX.contains(WORLD_X,WORLD_Y):
        ORIENTATION = RIGHT
    else:
        print("Error Cannot find proper Orientation: ")
        return

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


"""Call this from main.py to inject all shared components at once."""
def init_controller(wakeup, to_cv, cv_done, stop, index, lock, blocks):
    global controller_wakeup, controller_to_opencv, opencv_done, controller_stop
    global lookup_index, map_lock, active_blocks

    controller_wakeup = wakeup
    controller_to_opencv = to_cv
    opencv_done = cv_done
    controller_stop = stop
    lookup_index = index
    map_lock = lock
    active_blocks = blocks

"""The thread target. No arguments needed; it uses the globals directly!"""
def Controller_main():
    print("[Controller] Thread started and successfully.")

    # rbt = RobotClient(host=HOST, port=PORT)
    # connect_and_home(rbt)

    try:
        while not controller_stop.is_set():
            # --- wait idle ---
            controller_wakeup.wait()
            print("controller Wakeup")
            # make sure we are not stopping
            if controller_stop.is_set():
                break
            # --- request starting scan ---
            print("controller requesting fresh scan from openCV")
            controller_to_opencv.set()      # wake up opencv task
            opencv_done.wait()              # wait till done
            opencv_done.clear()             # clear signal so opencv pause
            # --- make sure workspace is not empty ---
            # --- pickup ---
            # 1. Grab a quick snapshot list of the IDs while locked
            with map_lock:
                # Converting to a list creates a copy of the keys instantly (very fast)
                id_snapshot = list(active_blocks.keys())
            # 2. Loop through the snapshot completely UNLOCKED
            for block_id in id_snapshot:
                # --- Tell your OpenCV task to check or do other work here ---
                # (Since the map isn't locked here, your OpenCV thread can write to it freely)
                # 3. Lock briefly ONLY when reading and printing the block
                with map_lock:
                    # Check if the block still exists (in case OpenCV deleted it)
                    lookup_index = block_id
                    if block_id in active_blocks:
                        block = active_blocks[block_id]
                        print(f"Found Block ID: {block_id} | Color: {block.color}")

                print("Controller requesting verification")
                controller_to_opencv.set()      # wake up opencv task
                opencv_done.wait()              # wait till done
                opencv_done.clear()             # clear signal so opencv pause

            with map_lock:
                active_blocks.clear()
                print("Shared map has been safely cleared.")
            lookup_index = -1
            print("Controller Done.")
            controller_wakeup.clear() # clear so sleeps next loop
    finally:
        # safe clean up
        # rbt.close()
        print("Controller Connection closed.")