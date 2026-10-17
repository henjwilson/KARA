import time

from parol6 import RobotClient

# The pneumatic gripper is binary (open/closed), driven by a digital output.
with RobotClient(host="127.0.0.1", port=5001) as rbt:
    rbt.select_tool("PNEUMATIC", "vertical")
    rbt.tool.open(wait=True)
    time.sleep(1.0)
    rbt.tool.close(wait=True)
    time.sleep(1.0)
    rbt.tool.open(wait=True)
