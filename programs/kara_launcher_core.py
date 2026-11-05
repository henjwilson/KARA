"""Pure configuration and command building for the KARA Programs launcher.

No robot code is imported here: users can inspect and test the launcher
without connecting to hardware or installing the robot backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_HOST = '127.0.0.1'
ROBOT_PORT = 5001


@dataclass(frozen=True)
class Program:
    key: str
    title: str
    path: str
    description: str
    group: str
    flags: tuple[str, ...] = ()
    camera: bool = False
    requires_robot: bool = True
    moves_robot: bool = True
    interactive: bool = False
    caution: str = ''


PROGRAMS = (
    Program('camera_station', 'Camera Station', 'programs/camera_station.py',
            'Live camera with jogging, calibration, drop calibrate, seek yellow, exposure and focus.',
            'Vision', camera=True,
            caution='Keys go to its camera window. T T recalibrates the table; M M runs drop calibrate.'),
    Program('seek_yellow', 'Seek Yellow · PICK', 'OpenCV/seek_yellow.py',
            'Continuously find yellow blocks, grip them, return home and release.', 'Vision', camera=True,
            caution='Requires a fixed calibrated camera and at least 6 valid drop-calibration pairs.'),
    Program('seek_yellow_dry', 'Seek Yellow · DRY RUN', 'OpenCV/seek_yellow.py',
            'Camera detection and motion planning ONLY. No motion commands.', 'Vision',
            flags=('--dry-run',), camera=True, requires_robot=False, moves_robot=False),
    Program('track_blocks', 'Track Yellow / Table', 'OpenCV/main.py',
            'View ArUco markers, detect blocks and display calibrated XY.', 'Vision',
            camera=True, requires_robot=False, moves_robot=False),
    Program('calibrate_camera', 'Calibrate Camera', 'OpenCV/main.py',
            'Record four marker positions for pixel-to-robot mapping.', 'Vision',
            flags=('--calibrate',), camera=True, requires_robot=False, moves_robot=False,
            caution='Drop-calibration pairs carry over only if the markers have not moved.'),
    Program('drop_calibrate', 'Collect Drop Calibration', 'programs/drop_calibrate.py',
            'Robot-guided grid measurement for accurate block pick locations.', 'Vision', camera=True,
            caution='Requires a yellow block in the gripper; camera-window keyboard controls.'),
    Program('refine_camera', 'Refine Camera Map', 'OpenCV/main.py',
            'Fit camera-to-robot map from stored coordinate pairs (edits calibration JSON).',
            'Vision', flags=('--refine',), requires_robot=False, moves_robot=False,
            caution='Replaces the current map; drop-calibration pairs carry over.'),
    Program('commander', 'Commander · XY', 'programs/Commander.py',
            'Command input: x,y  |  home  |  park  |  quit. Homes/parks on launch.',
            'Interactive', interactive=True),
    Program('xyz_console', 'Cartesian XYZ Console', 'programs/xy_jog.py',
            'Command input: x y z [rx ry rz]  |  pose  |  home  |  quit.',
            'Interactive', interactive=True),
    Program('six_axis_console', '6D Pose Console', 'programs/workspace_mapping/self_move.py',
            'Command input: x,y,z,rx,ry,rz  |  home  |  quit.',
            'Interactive', interactive=True),
    Program('orientation_console', 'Pickup Orientation', 'programs/workspace_mapping/join_space.py',
            'Command input: x,y,T/L/R/B/F  |  home  |  quit. Uses fixed low Z.',
            'Interactive', interactive=True, caution='Uses a hardcoded pickup height below the table zero; verify clearance.'),
    Program('cartesian_keyboard', 'Cartesian Keyboard', 'programs/cartesian_keyboard.py',
            'WASDQE for XYZ, H=home, Space=stop, Esc=exit; pynput controls.',
            'Interactive', caution='This uses GLOBAL keyboard events, not the command entry box.'),
    Program('joint_keyboard', 'Joint Keyboard', 'programs/Aliou-Practice.py',
            'Keyboard jog one joint at a time; H=home and Esc=exit.',
            'Interactive', caution='This uses GLOBAL keyboard events, not the command entry box.'),
    Program('gripper_test', 'Gripper Open/Close Test', 'programs/gripper_test.py',
            'Open → close → open pneumatic gripper.', 'Demos'),
    Program('zigzag', 'Zigzag Scan', 'programs/zigzag_scan.py',
            'Run a multirow zigzag scanning trajectory.', 'Demos'),
    Program('circle', 'Draw Circles', 'programs/draw_circle.py',
            'Draw three circles and a sine-wave spline.', 'Demos'),
    Program('precision', 'Precision / Pencil', 'programs/precision.py',
            'Tool offset, pencil handling and tool-frame rotations.', 'Demos',
            caution='Requires the physical pencil/fixture arrangement used by the script.'),
    Program('demo_showcase', 'Motion Showcase', 'programs/demo_showcase.py',
            'Long demonstration of movement primitives and gripper actions.', 'Demos',
            caution='Uses hardcoded tool and pencil motion paths.'),
    Program('speed_comparison', 'Speed Comparison', 'programs/speed_comparison.py',
            'Runs a trajectory at several speeds and motion profiles.', 'Demos'),
    Program('map_workspace', 'Map Workspace', 'programs/map_workspace.py',
            'Explore a grid of reachability targets; can take a long time.', 'Demos',
            caution='Moves through many points in a large grid; check workspace first.'),
    Program('cycle_start', 'PLC I/O Cycle', 'programs/cycle_start.py',
            'Waits on digital input and runs three preprogrammed cycles.', 'Demos',
            caution='Depends on PLC wiring / digital input and output states.'),
)
PROGRAM_BY_KEY = {p.key: p for p in PROGRAMS}


SERVER_COMMANDS = {
    'PAROL6 Robot Server': 'parol6-server',
    'Waldo Commander Web UI': 'waldo-commander',
}


def locate_python() -> str:
    """Prefer a project virtualenv when the launcher was opened via system Python."""
    if sys.prefix != getattr(sys, 'base_prefix', sys.prefix):
        return sys.executable
    for env in ('.venv', 'venv'):
        for rel in ('Scripts/python.exe', 'bin/python'):
            p = PROJECT_ROOT / env / rel
            if p.is_file():
                return str(p)
    return sys.executable


def find_server_command(name: str, python: str) -> list[str] | None:
    """Find an installed console script without relying on a shell or PATH hacks."""
    if name not in SERVER_COMMANDS:
        raise KeyError(name)
    command = SERVER_COMMANDS[name]
    # Do not use resolve(): a virtualenv python may be a symlink to system Python.
    python_path = Path(os.path.abspath(python))
    for folder in (python_path.parent, python_path.parent / 'Scripts', python_path.parent.parent / 'bin'):
        for suffix in ('.exe', '', '.cmd') if os.name == 'nt' else ('',):
            candidate = folder / (command + suffix)
            if candidate.is_file():
                return [str(candidate)]
    found = shutil.which(command)
    return [found] if found else None


def program_command(key: str, python: str, camera: int = 0) -> list[str]:
    program = PROGRAM_BY_KEY[key]
    if not 0 <= camera <= 20:
        raise ValueError('Camera index must be between 0 and 20')
    target = PROJECT_ROOT / program.path
    if not target.is_file():
        raise FileNotFoundError(target)
    cmd = [python, '-u', str(target), *program.flags]
    if program.camera:
        cmd += ['--camera', str(camera)]
    return cmd


def action_command(action: str, python: str) -> list[str]:
    if action not in ('home', 'park', 'pose', 'gripper_open', 'gripper_close', 'stop'):
        raise ValueError('Unknown robot action')
    return [python, '-u', str(PROJECT_ROOT / 'programs' / 'kara_robot_action.py'), action]
