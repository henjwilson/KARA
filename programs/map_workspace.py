"""
Workspace mapper for the PAROL6 robot arm.

Sweeps a grid of XY points (at a fixed Z / orientation) against the REAL
running parol6-server and records which ones actually succeed. This gives
you ground-truth reachable bounds instead of guessed ones.

Setup:
    Run `parol6-server` in a separate terminal FIRST, then run this script.

Usage:
    python3 map_workspace.py
    (edit the CONFIG section below to change range/resolution)
"""

import time
from parol6 import RobotClient
from parol6.utils.errors import MotionError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 5001
TOOL = "SSG-48"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

Z, RX, RY, RZ = 150, 90, 0, 90

# Grid to sweep - tune this. Wider range / smaller step = slower but more precise.
X_RANGE = range(-350, 351, 25)
Y_RANGE = range(0, 451, 25)

MOVE_SPEED = 0.8  # fast, since we just care about reachability, not smoothness
REHOME_EVERY = 40  # re-home every N attempts to avoid drifting into a weird config


def connect_and_home(rbt: RobotClient) -> None:
    rbt.select_tool(TOOL)
    rbt.tool.calibrate()
    current = rbt.angles()
    if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
        rbt.home(wait=True)


def main() -> None:
    rbt = RobotClient(host=HOST, port=PORT)
    connect_and_home(rbt)
    print(f"Sweeping X {X_RANGE.start}..{X_RANGE.stop - 1}, Y {Y_RANGE.start}..{Y_RANGE.stop - 1} (Z={Z} fixed)")

    results = {}
    attempts = 0

    try:
        for y in Y_RANGE:
            for x in X_RANGE:
                pose = [x, y, Z, RX, RY, RZ]
                try:
                    rbt.move_j(pose=pose, speed=MOVE_SPEED)
                    results[(x, y)] = True
                except MotionError:
                    results[(x, y)] = False
                except Exception:
                    results[(x, y)] = False

                attempts += 1
                if attempts % REHOME_EVERY == 0:
                    rbt.home(wait=True)

        rbt.home(wait=True)

        # --- Print results as a text map ---
        xs = list(X_RANGE)
        ys = list(Y_RANGE)
        print()
        print("     " + "".join(f"{x:>5}" for x in xs))
        for y in reversed(ys):
            row = "".join(("  OK " if results[(x, y)] else "  .  ") for x in xs)
            print(f"{y:>4} {row}")

        # --- Print a suggested bounding box ---
        ok_pts = [pt for pt, ok in results.items() if ok]
        if ok_pts:
            xs_ok = [p[0] for p in ok_pts]
            ys_ok = [p[1] for p in ok_pts]
            print()
            print(f"Reachable X range found: {min(xs_ok)} to {max(xs_ok)}")
            print(f"Reachable Y range found: {min(ys_ok)} to {max(ys_ok)}")
            print("(Note: this is a rectangle around an irregular region - some")
            print(" points inside this box may still be unreachable. Check the")
            print(" map above, and pull bounds in from the printed edges for a")
            print(" safety margin.)")
        else:
            print("No reachable points found - check Z/orientation config.")

    finally:
        rbt.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()