"""What is on the table, in 3D, and where the next cube of a build goes.

Cubes are kept in robot XY with a stacking level (0 on the table). The camera
sees only the top of a stack, so the cubes under it are filled in as ``hidden``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cube_pose import CUBE_SIDE_MM

# Cubes closer than this (mm) in XY are in the same column.
SAME_COLUMN_MM = 6.0
# Room (mm) a column needs from its neighbours for the gripper's jaws.
GRIPPER_CLEARANCE_MM = 30.0


@dataclass
class Cube:
    x: float
    y: float
    level: int
    yaw: float
    color: str = "hidden"


@dataclass
class Slot:
    x: float
    y: float
    level: int
    yaw: float
    support: list[tuple[float, float]]  # column XY under it; empty on the table

    @property
    def z(self) -> float:
        return self.level * CUBE_SIDE_MM


class Workspace:
    def __init__(self, side_mm: float = CUBE_SIDE_MM):
        self.side = side_mm
        self.cubes: list[Cube] = []

    def update(self, seen: list[dict]) -> None:
        """Replace the contents with the tops seen, each a dict of x, y, level,
        yaw and color."""
        self.cubes = []
        for top in seen:
            self.cubes.append(Cube(top["x"], top["y"], top["level"], top["yaw"], top["color"]))
            self.cubes += [Cube(top["x"], top["y"], level, top["yaw"])
                           for level in range(top["level"])]

    def column_height(self, x: float, y: float) -> int:
        """Number of cubes standing at (x, y)."""
        here = [c.level + 1 for c in self.cubes if math.dist((c.x, c.y), (x, y)) <= SAME_COLUMN_MM]
        return max(here, default=0)

    def towers(self) -> list[tuple[float, float, int]]:
        """Each column of more than one cube, as (x, y, cubes)."""
        return [(c.x, c.y, c.level + 1) for c in self.cubes if c.level > 0 and c.color != "hidden"]

    def is_clear(self, x: float, y: float) -> bool:
        """True if the gripper has room to work at (x, y)."""
        return all(math.dist((c.x, c.y), (x, y)) > GRIPPER_CLEARANCE_MM
                   or math.dist((c.x, c.y), (x, y)) <= SAME_COLUMN_MM for c in self.cubes)

    def tower(self, x: float, y: float, height: int, yaw: float = 0.0) -> list[Slot]:
        return [Slot(x, y, level, yaw, [(x, y)] if level else []) for level in range(height)]

    def pyramid(self, x: float, y: float, base: int, yaw: float = 0.0,
                gap_mm: float = 1.0) -> list[Slot]:
        """Slots of a stepped wall: ``base`` cubes in a row along ``yaw``, then
        one fewer on each level, every cube bridging the two below it. Rows go
        on from the center so the arm never reaches past a placed cube."""
        pitch = self.side + gap_mm
        along = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
        slots = []
        for level in range(base):
            count = base - level
            for index in range(count):
                offset = (index - (count - 1) / 2.0) * pitch
                sx, sy = x + along[0] * offset, y + along[1] * offset
                support = [] if level == 0 else [
                    (sx + along[0] * side * pitch / 2.0, sy + along[1] * side * pitch / 2.0)
                    for side in (-1, 1)]
                slots.append(Slot(sx, sy, level, yaw, support))
        return slots

    def _filled(self, slot: Slot) -> bool:
        return self.column_height(slot.x, slot.y) > slot.level

    def _bridged(self, slot: Slot) -> bool:
        """Its supports are only as tall as the level it sits on."""
        return all(self.column_height(x, y) >= slot.level for x, y in slot.support)

    def next_slot(self, slots: list[Slot]) -> Slot | None:
        """The first unfilled slot with its supports in place, or None when the
        build is finished or can't go on."""
        for slot in slots:
            if not self._filled(slot):
                return slot if self._bridged(slot) else None
        return None
