# Table vision → robot coordinates

This program detects ArUco markers and colored blocks in a fixed camera view. It
uses the known robot WRF coordinates of marker centers to calibrate a planar
pixel-to-robot transform, then shows block positions in millimeters and a
top-down virtual table map.

## Install and calibrate

From the repository root:

```bash
python3 -m venv OpenCV/venv
source OpenCV/venv/bin/activate
python -m pip install -r OpenCV/requirements.txt
python OpenCV/main.py --calibrate --camera 0
```

Put ArUco markers **2, 5, 8, and 11** flat on the table and spread them around
the work area. The default dictionary is `DICT_4X4_50`; change
`--dictionary` if your markers use a different dictionary. In the camera
window press **C** when all four IDs are visible. Their robot WRF coordinates
are preset in `main.py` from the readings below, in millimeters:

| ArUco ID | X | Y | Z |
|---:|---:|---:|---:|
| 2 | 299.503 | 199.626 | -2.163 |
| 5 | 182.910 | 202.169 | -11.766 |
| 8 | 321.658 | -254.597 | 7.108 |
| 11 | 215.268 | -261.558 | -8.832 |

These were touched with the `PNEUMATIC` tool, `vertical` variant (the gripper
all the programs select), with the tip held just above each marker. The table
surface is taken to be `TOUCH_GAP_MM` (2 mm, in `main.py`) below them; a
plane is fitted through the four heights because the table isn't level in
robot coordinates.

The repeated readings for each point were averaged; each is the TCP touching
the center of its marker. Seen from the camera, ID 8 is top-left, 11
top-right, 5 bottom-right, and 2 bottom-left, so image-up is robot -Y and
image-right is robot -X. Because these four points are nearly symmetric, the
camera can't tell this pairing from its 180°-rotated twin; calibration prints
a NOTE when that happens. If the robot goes to the opposite side of the table,
the pairing is rotated: swap the coordinates of IDs 2 and 11 and of 5 and 8.

### How calibration keeps the map realistic

Pressing **C** averages the marker corners over 15 frames, then fits the
camera-to-robot mapping in two parts:

- **Shape comes from the camera.** Every ArUco marker is a printed square of
  the same size, so the mapping must turn all 16 detected corners into equal
  squares. This fixes perspective, aspect ratio and skew without trusting the
  touched points' accuracy.
- **Placement comes from the robot.** The touched centers only position,
  rotate and scale that shape (least squares), so touch errors average out
  instead of stretching or shearing the map. Mirrored maps are rejected.
- **Mixed-up IDs are caught.** All 24 ways of pairing the four IDs with the
  four robot coordinates are tried. If another pairing fits far better than
  `MARKER_ROBOT_COORDS_MM`, it prints which IDs look swapped and uses the
  corrected pairing.

Calibration prints how far the camera puts each marker from its touched point,
and the marker size it measured; compare that with a ruler. If you know the
printed size, pass it for a slightly better fit, e.g. `--marker-size 19`. This
is the black square only: a 4x4 marker printed with its usual one-cell white
margin is 6/8 of the paper square (2.5 cm paper -> about 19 mm black). Both `main.py` and `seek_yellow.py` accept
`--calibrate` and share the same `table_calibration.json`, and both warn if a
marker has moved in the image since calibrating.

The camera must remain fixed after calibration. Tracking must use the same
camera resolution and marker dictionary. To use another camera or resolution,
run calibration again with the same `--camera`, `--width`, and `--height`
options you will use for tracking.

### Refine with blocks the robot touched

After calibrating, the map can be corrected using blocks measured both ways.
In `programs/cartesian_keyboard.py`, for each block (spread them over the
board): press **Y** (camera saves the block's top center as a `camera` row),
then jog the TCP to the block's top center and press **C** (a `robot` row).
Or let the robot place the block itself: `programs/drop_calibrate.py` visits
a grid inside the markers with the block in the gripper, you lower it to the
table with Q / E and press Enter, and it releases, homes, logs the camera and
release positions, and picks the block back up for the next point. Then run:

```bash
.venv/bin/python OpenCV/main.py --refine
```

Each camera row is paired with the nearest robot row, and a rotation + scale +
shift taking camera XY onto robot XY is fitted and folded into
`table_calibration.json`. It prints each pair's error before and after, plus
the error on points left out of the fit (an honest estimate), and refuses to
save a correction that doesn't help. Only camera rows taken since the
calibration last changed are used, so to refine again, capture new pairs.
Running `--calibrate` again starts over from the markers and drops any
refinement.

## Track blocks

```bash
python OpenCV/main.py --camera 0
```

The camera window overlays marker IDs and detected colored blocks. The second
window shows their calibrated robot WRF XY positions on a uniformly scaled
millimeter grid. Yellow blocks are only detected inside the border formed by
the four calibration markers (drawn in blue); `--border-margin` (default 30 px)
lets a block center sit slightly outside it. Each block is fitted with a
rotated rectangle, and regions that fill less than `--min-fill` (default 0.6)
of their rectangle are ignored as non-rectangular.
To suppress noise, regions smaller than `--min-area` (default 400 px) are
ignored, and a block is only reported once it has been seen in
`--confirm-frames` (default 3) consecutive frames at about the same spot.
The camera overlay shows the number of detected yellow regions.
Press **P** to print current XYZ coordinates in the terminal; **Z** is
estimated from the calibrated table-height plane. Press **Q** to quit.

The camera sees a block's top and some of its sides. The top face, the
brightest part of the yellow region when lit from above, is outlined and its
center marked with a white dot. The reported XY is the table point directly
below that center. The visible region is the top face plus its own copy
shifted down to the bottom face, so the table point is found without knowing
the block's height. If no top face stands out (e.g. glare), the outline is
orange, labelled "no top face", and the lowest edge of the region is used
instead. Accuracy depends on
the marker coordinate mapping, camera stability, lighting, and blocks resting
on the same plane as the markers. This program only reports coordinates; it
does not send motion commands to the robot.

Useful options:

```bash
python OpenCV/main.py --help
python OpenCV/main.py --calibrate --camera 1 --dictionary 5X5_100
python OpenCV/main.py --camera 1 --dictionary 5X5_100 --min-area 700 --confirm-frames 5
```

## Pick up yellow blocks

`seek_yellow.py` keeps watching for yellow blocks inside the ArUco border.
When one has been seen steadily (`--confirm-frames`, default 10), the robot
picks it up and drops it at home, then looks for the next block. The target
XY is the median over those frames, using only frames where the block's top
face was found; a block whose top face shows in fewer than half of them is
labelled "no top face, waiting" and isn't picked until it does. Blocks it
already tried are drawn grey and skipped until they are removed; out-of-reach
blocks are skipped too. Press **Q** to stop.

It relies on the pairs `programs/drop_calibrate.py` logged with the current
camera calibration (at least 6; it refuses to start without them):

- **Position:** a smoothed thin-plate spline maps the camera's XY onto where
  the robot actually released the block, which also absorbs the arm's own
  positioning error. On start it prints the camera's average error before
  and the error after, measured on pairs left out of the fit.
- **Height:** the pick height is interpolated from the release heights, the
  TCP heights at which the block sat on the table.
- **Orientation:** planned as in `drop_calibrate.py` (`OpenCV/placement.py`):
  Rx 90 with at least 45 deg of Ry freedom, Rx lowered only where needed.

Don't run `main.py --refine` after collecting pairs: it changes the map, so
the pairs no longer apply. Recalibrating means collecting them again.

For each block the arm moves 50 mm above it, descends straight down, closes
the gripper, lifts, homes, and opens the gripper to drop the block.

It needs both `parol6` and OpenCV, so run it from the repo's `.venv` with
`parol6-server` already running:

```bash
.venv/bin/python OpenCV/seek_yellow.py --camera 0
.venv/bin/python OpenCV/seek_yellow.py --camera 0 --dry-run   # no robot motion
```
