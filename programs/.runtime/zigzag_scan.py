from parol6 import RobotClient

rbt = RobotClient(host='127.0.0.1', port=5001)

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

# Select tool, and home only if not already near the home pose
rbt.select_tool("SSG-48")
rbt.tool.calibrate()
current = rbt.angles()
if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
    rbt.home()

# move_j — TCP follows a curved arc through joint space
rbt.move_j(pose=[100, 340, 334, 90, 0, 90], speed=0.5)

# move_l — TCP travels in a straight Cartesian line
rbt.move_l([-50, 340, 334, 90, 0, 90], speed=0.5)
