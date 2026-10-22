import threading
import time
import random
from models import Block

rbt = None
controller_to_opencv = None
opencv_done = None
opencv_stop = None
# lock
map_lock =  None
# map
active_blocks = None
# -1 mean replace all. any other means get that id
lookup_index  = None

"""Call this from main.py to inject all shared components at once."""
def init_opencv(to_cv, done, stop, index, lock, blocks):
    global controller_to_opencv, opencv_stop, opencv_done
    global lookup_index, map_lock, active_blocks

    controller_to_opencv = to_cv
    opencv_done = done
    opencv_stop = stop
    lookup_index = index
    map_lock = lock
    active_blocks = blocks

def opencv_main():
    print("[Opencv] Thread started and successfully.")
    colors = ["red", "blue", "green", "yellow", "orange"]
    #--- wait idle --
    try:
        while not opencv_stop.is_set():
            #--- wait idle --
            controller_to_opencv.wait()
            # 2. Immediately close the gate so it blocks on the NEXT loop pass
            controller_to_opencv.clear()

            print("Opencv wakeup")
            if opencv_stop.is_set():
                break

            with map_lock:
                print("Opencv has lock")
                if lookup_index == -1:
                    for _ in range(5):
                        # 1. Generate a unique ID (or use a counter)
                        block_id = random.randint(1000, 9999)

                        # 2. Instantiate the Block with random values
                        new_block = Block(
                            id=block_id,
                            color=random.choice(colors),
                            x=round(random.uniform(0.0, 100.0), 2),
                            y=round(random.uniform(0.0, 100.0), 2),
                            angle=round(random.uniform(0.0, 360.0), 1),
                            confidence=round(random.uniform(0.7, 1.0), 2)
                        )
                        # 3. Add it to your shared map using the ID as the key
                        active_blocks[block_id] = new_block
                elif lookup_index in active_blocks:
                    # 2. Grab the Block dataclass object
                    found_block = active_blocks[lookup_index]

                    # 3. Print out its key and its internal data
                    print(f"Found Block Key (ID): {lookup_index}")
                    print(f"Color: {found_block.color}, Confidence: {found_block.confidence}")
                else:
                    print(f"Block with ID {lookup_index} not found in the workspace.")

            print("Opencv Done:")
            opencv_done.set()
    finally:
        # safe clean up
        # rbt.close()
        print("Opencv Connection closed.")
