import time
from parol6 import RobotClient

rbt = RobotClient()

print("Moving to home position...")
rbt.home()
# x, y, z, rx, ry, rz
# 90.0, -90.0, 180.0, 0.0, 0.0, 180.0
pose = [90.0, -90.0, 180.0, 0.0, 0.0, 90]
speed = 0.3
rbt.move_l([90.00, -90.00, 180.00, 0.00, 0.00, 90.00], speed=0.2, accel=0.2)  # Recording start position
# rbt.move_l(pose, speed=speed)
