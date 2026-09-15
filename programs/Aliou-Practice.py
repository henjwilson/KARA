"""
Keyboard jog-practice script for the PAROL6 robot arm.

Setup:
    pip install pynput
    Run `parol6-server` in a separate terminal FIRST, then run this script.

Controls:
    Left / Right arrow  -> Joint 1  (base rotation)
    Up / Down arrow      -> Joint 2  (shoulder)
    Q / E                 -> Joint 3  (elbow)
    A / S                  -> Joint 4  (wrist)
    Z / X                  -> Joint 5  (wrist)
    C / V                  -> Joint 6  (tool rotate)
    H                       -> Go to HOME position
    Esc                     -> Stop motion and quit

Notes:
    - Uses pynput instead of the `keyboard` package: `keyboard` needs root
      and is unreliable on macOS. pynput works through macOS's Accessibility
      API without sudo.
    - macOS: grant your terminal app Accessibility permissions the first
      time you run this (System Settings -> Privacy & Security ->
      Accessibility), or key presses will be silently ignored.
    - Only one joint jogs at a time in this version (first held key wins).
      RobotClient.jog_j() also supports moving several joints at once via
      jog_j(joints=[...], speeds=[...]) if you want to extend this later.
"""

import time
from pynput import keyboard
from parol6 import RobotClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 5001
TOOL = "SSG-48"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

JOG_SPEED = 0.25   # fraction of each joint's max speed (0.0 - 1.0)
JOG_ACCEL = 1.0    # fraction of max accel (0.0 - 1.0)
LOOP_HZ = 20        # how often we poll held keys and refresh the jog command

# jog_j() only moves for `duration` seconds unless it's refreshed, so we make
# each call last slightly longer than one loop tick to avoid stutter between polls.
JOG_DURATION = (1.0 / LOOP_HZ) + 0.05

# Joint index is 0-based: J1=0, J2=1, J3=2, J4=3, J5=4, J6=5
# Each entry is (positive-direction key, negative-direction key)
JOINT_KEYS = {
    0: ("right", "left"),   # Joint 1
    1: ("up", "down"),      # Joint 2
    2: ("e", "q"),          # Joint 3
    3: ("s", "a"),          # Joint 4
    4: ("x", "z"),          # Joint 5
    5: ("v", "c"),          # Joint 6
}

HOME_KEY = "h"
QUIT_KEY = "esc"

# ---------------------------------------------------------------------------
# Keyboard state (updated by pynput's background listener thread)
# ---------------------------------------------------------------------------
pressed_keys: set[str] = set()
quit_requested = False


def _key_name(key) -> str | None:
    """Normalize a pynput key event to the lowercase string names used above."""
    if isinstance(key, keyboard.KeyCode):
        return key.char.lower() if key.char else None
    # Special keys (arrows, esc, etc.) - strip the "Key." prefix pynput adds
    name = str(key).replace("Key.", "")
    return name


def on_press(key):
    global quit_requested
    name = _key_name(key)
    if name is None:
        return
    if name == QUIT_KEY:
        quit_requested = True
        return False  # stop the listener
    pressed_keys.add(name)


def on_release(key):
    name = _key_name(key)
    if name is not None:
        pressed_keys.discard(name)


def connect_and_home(rbt: RobotClient) -> None:
    rbt.select_tool(TOOL)
    rbt.tool.calibrate()
    current = rbt.angles()
    if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
        print("Homing...")
        rbt.home(wait=True)
    print("Ready. Hold a mapped key to jog a joint. Press Esc to stop and quit.")


def main() -> None:
    rbt = RobotClient(host=HOST, port=PORT)
    connect_and_home(rbt)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    was_moving = False
    try:
        while not quit_requested:
            if HOME_KEY in pressed_keys:
                rbt.stop()
                was_moving = False
                print("Homing...")
                rbt.home(wait=True)
                pressed_keys.discard(HOME_KEY)
                time.sleep(0.2)
                continue

            active_joint = None
            direction = 0
            for joint_idx, (pos_key, neg_key) in JOINT_KEYS.items():
                if pos_key in pressed_keys:
                    active_joint, direction = joint_idx, 1
                    break
                if neg_key in pressed_keys:
                    active_joint, direction = joint_idx, -1
                    break

            if active_joint is not None:
                rbt.jog_j(active_joint, speed=JOG_SPEED * direction, duration=JOG_DURATION, accel=JOG_ACCEL)
                was_moving = True
            elif was_moving:
                # Key released since the last tick -> stop immediately
                rbt.stop()
                was_moving = False

            time.sleep(1.0 / LOOP_HZ)
    finally:
        rbt.stop()
        rbt.close()
        listener.stop()
        print("Stopped. Connection closed.")


if __name__ == "__main__":
    main()