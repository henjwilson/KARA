from parol6 import RobotClient

with RobotClient(host="127.0.0.1", port=5001) as rbt:
    print("Homing...")
    rbt.home(wait=True)
    print("Home.")
