"""Small one-shot actions for buttons in KARA Programs GUI.

The RobotClient API and the HOME/PARK constants are taken from existing scripts.
This file only connects when explicitly launched by the user.
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description='One-shot PAROL6 action from the KARA GUI')
    parser.add_argument('action', choices=('home', 'park', 'pose', 'gripper_open', 'gripper_close', 'stop'))
    args = parser.parse_args()

    from parol6 import RobotClient
    from Commander import PARK_ANGLES, MOVE_SPEED

    with RobotClient(host='127.0.0.1', port=5001) as robot:
        if args.action == 'home':
            print('Homing...', flush=True)
            robot.home(wait=True)
            print(f'Home complete. Pose: {robot.pose()}', flush=True)
        elif args.action == 'park':
            print('Moving to Commander.py PARK_ANGLES...', flush=True)
            robot.move_j(pose=PARK_ANGLES, speed=MOVE_SPEED, wait=True)
            print(f'Park complete. Pose: {robot.pose()}', flush=True)
        elif args.action == 'pose':
            print(f'Joint angles: {robot.angles()}', flush=True)
            print(f'Cartesian pose: {robot.pose()}', flush=True)
        elif args.action in ('gripper_open', 'gripper_close'):
            robot.select_tool('PNEUMATIC', 'vertical')
            if args.action == 'gripper_open':
                robot.tool.open(wait=True)
            else:
                robot.tool.close(wait=True)
            print(f'Gripper {args.action.split("_")[1]} complete.', flush=True)
        else:
            robot.stop()
            print('Stop request sent to backend (not a hardware emergency stop).', flush=True)


if __name__ == '__main__':
    main()
