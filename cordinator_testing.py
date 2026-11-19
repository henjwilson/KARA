
from dataclasses import dataclass
import ast


@dataclass
class area:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class Box:
    def __init__(self, message):
        self.x, self.y, self.color, self.angle = message


class Coordinator:

    def coordinate_zone(self, box: Box):

        self.ZONE_A = area(220, 130, 290, 280)
        self.ZONE_B = area(226, -150, 290, 130)
        self.ZONE_C = area(200, -280, 290, -150)
        self.ZONE_D = area(120, 130, 219, 280)
        self.ZONE_E = area(0, 0, 0, 0)
        self.ZONE_F = area(150, -100, 200, 0)
        self.ZONE_G = area(150, -280, 200, -100)
        self.ZONE_G2 = area(120, -280, 150, -150)
        self.ZONE_H = area(120, 0, 150, 129)
        self.ZONE_I = area(120, -150, 150, 149)

        rx, ry, rz = 0, 0, 0
        def lerp(x, x1, y1, x2, y2):
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

        if self.ZONE_B.contains(box.x, box.y):

            if not -45 <= box.angle <= 45:
                raise ValueError(
                    "Box angle must be between -45 and 45 degrees"
                )

            angle = abs(box.angle)

            rx = lerp(angle, 0, 45, 45, 85)
            ry = lerp(angle, 0, -75, 45, -40)
            rz = lerp(angle, 0, 50, 45, 90)

            if box.angle < 0:
                rx = -rx
                rz = -rz

        elif box.y >= 0:

            rx = 75
            if self.ZONE_A.contains(box.x, box.y):
                ry = box.angle
                if box.angle > 0:
                    ry = -box.angle - 45

            elif self.ZONE_D.contains(box.x, box.y):
                ry = round(box.angle)

            elif self.ZONE_E.contains(box.x, box.y):
                raise ValueError("Cannot solve Zone E right now")
            if ry >= 0:
                rz = 90
            elif ry >= -20:
                rz = lerp(ry, 0, 90, -20, 80)
            elif ry >= -45:
                rz = 80
            elif ry >= -90:
                rz = lerp(ry, -45, 80, -90, 70)

            if self.ZONE_H.contains(box.x, box.y):
                ry = box.angle
                rz = 100
                if box.angle < 0:
                    ry = -box.angle + 45

            if rx is 0:
                raise ValueError("Cannot Find Zone")

        elif box.y < 0:

            rx = -75
            if self.ZONE_C.contains(box.x, box.y):
                ry = box.angle
                if box.angle > 0:
                    ry = -box.angle - 45

            elif self.ZONE_F.contains(box.x, box.y):
                print("zone F")
                if -30 <= box.angle <= 30:
                    ry = round(box.angle)
                else:
                    raise ValueError(
                        "Angle too big for Zone F (-30 to 30)"
                    )

            elif self.ZONE_G.contains(box.x, box.y):
                ry = box.angle

            elif self.ZONE_G2.contains(box.x, box.y):
                ry = box.angle

            elif self.ZONE_I.contains(box.x, box.y):
                print("zone I")
                ry = box.angle
                if box.angle < 0:
                    ry = -box.angle + 45
            else:
                raise ValueError("Cannot Find Zone")

            rz = lerp(ry, 0, -90, 45, -110)


        return round(rx), round(ry), round(rz)


# -------------------------------
# TERMINAL TEST
# -------------------------------

def main():
    coordinator = Coordinator()

    print("=== Coordinator Tester ===")
    print("Enter: [x, y, 'color', angle]")
    print("Example: [250, -50, 'red', 45]")
    print("Type 'q' to quit.\n")

    while True:
        user_input = input("Box > ").strip()

        if user_input.lower() in ("q", "quit", "exit"):
            break

        try:
            values = ast.literal_eval(user_input)

            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError("Use [x, y, 'color', angle]")

            box = Box(values)

            rx, ry, rz = coordinator.coordinate_zone(box)

            print(f"Input: {values}")
            print(f"Output: rx={rx}, ry={ry}, rz={rz}\n")

        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
