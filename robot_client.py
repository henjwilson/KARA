"""Two-port, single-threaded TCP coordinator SHELL.

Port 1 speaks newline-delimited JSON. Port 2's protocol is intentionally
undefined: replace the two adapter functions when you know its format.

The coordinator is a TCP CLIENT to two independent servers. It does not bind
or listen on either port. Run servers separately when testing.

Important: This is a framework, not a robot safety system. Hardware emergency
stops must work independently of TCP or this Python process.
"""

import json
import logging
import selectors
import socket
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from parol6 import RobotClient
from parol6.utils.errors import MotionError

# ============================================================================
# GLOBAL CONFIGURATION -- change these when testing.
# These are the SERVER addresses/ports that this client connects TO.
# ============================================================================

# ------------------------------------------------------------------------
# ROBOT VARIABLE:
# ------------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 5001
TOOL = "PNEUMATIC"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

PARK_ANGLES = [0,250,45,90,0,90]
PARK_TOLERANCE_DEG = 2.0

MOVE_SPEED = 0.25

# ------------------------------------------------------------------------
# WALDO-SERVER VARIABLE
# ------------------------------------------------------------------------
# WALDO_CONNECTED = False

# ------------------------------------------------------------------------
# PORT1 VARIABLES:
# ------------------------------------------------------------------------
PORT1_HOST = "127.0.0.1"
PORT1_PORT = 5002
PORT1_CONNECTED = False
# When disconnected, retry without holding up the other connection.
RECONNECT_INTERVAL_SEC = 2.0
# How long an unfinished TCP connection attempt may remain pending.
CONNECT_TIMEOUT_SEC = 5.0
# Upper bound on a single incoming newline-delimited message.
MAX_MESSAGE_BYTES = 1024 * 1024
# For tests, logs show connections, received messages, and state changes.
LOG_LEVEL = logging.INFO

PICK_UP_Z = 15

logger = logging.getLogger("coordinator")

# z = x+10
# Z/x multiples of 5

class State(Enum):
    IDLE = auto()             # Ready to accept one new request from Port 1.
    PROCESSING = auto()       # Doing the short local processing step.
    WAITING_FOR_ACK = auto()  # Port 2 has the task; awaiting its completion.
    FAULT = auto()            # A fault occurred; recovery policy decides next step.


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

@dataclass
class Connection:
    """Per-server TCP state; buffers are separate because TCP is a byte stream."""
    name: str
    host: str
    port: int
    sock: Optional[socket.socket] = None
    connected: bool = False
    connecting: bool = False
    connect_started: float = 0.0
    next_retry: float = 0.0
    incoming: bytearray = field(default_factory=bytearray)
    outgoing: bytearray = field(default_factory=bytearray)


class Coordinator:

    def __init__(self):
        # monitors both sockets and notify when one updates
        self.selector = selectors.DefaultSelector()
        self.port1 = Connection("PORT1", PORT1_HOST, PORT1_PORT)
        self.state = State.IDLE
        self.emergency_active = False
        self.emergency_reason = None
        self.current_fault = None
        self.active_request_id = None
        self.running = True
        self.WALDO_CONNECTED = False
        self.WALDO_PARK = False
        self.rbt = None


        # x_min, y_min, x_max,y_max
        self.LEFT_AREA = area(100,135,260,250)
        # x_min, y_min, x_max,y_max
        self.CENTER_AREA = area(100,-134,260,134)
    # ------------------------------------------------------------------------
    # SOCKET CONNECTION MANAGEMENT
    # ------------------------------------------------------------------------
    def try_connect(self, conn):
        """Begin a NONBLOCKING connect, so one offline server cannot freeze us."""
        if conn.sock is not None or time.monotonic() < conn.next_retry:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)
            result = sock.connect_ex((conn.host, conn.port))
            # connect_ex may succeed immediately (0), or still be in progress.
            # The selector reports writability when the connection completes.
            conn.sock = sock
            conn.connecting = True
            conn.connect_started = time.monotonic()
            self.selector.register(sock, selectors.EVENT_WRITE, conn)
            logger.info("%s: connecting to %s:%s (code=%s)",
                        conn.name, conn.host, conn.port, result)
        except OSError as exc:
            logger.info("%s: connect failed: %s", conn.name, exc)
            self.disconnect(conn)

    def finish_connect(self, conn):
        """Verify that a writable connecting socket actually connected."""
        error = conn.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if error:
            # print("%s: connection refused/failed (code=%s)", conn.name, error)
            self.disconnect(conn)
            return
        conn.connecting = False
        conn.connected = True
        logger.info("%s: connected", conn.name)
        self.update_interest(conn)
        # A new connection does not automatically clear an emergency or fault.

    def update_interest(self, conn):
        """Read always; request write events only when bytes are queued."""
        if conn.sock is None or conn.connecting:
            return
        events = selectors.EVENT_READ
        if conn.outgoing:
            events |= selectors.EVENT_WRITE
        self.selector.modify(conn.sock, events, conn)

    def disconnect(self, conn):
        """Close a given connection; leave the OTHER connection running."""
        was_connected = conn.connected
        if conn.sock is not None:
            try:
                self.selector.unregister(conn.sock)
            except (KeyError, ValueError):
                pass
            conn.sock.close()
        conn.sock = None
        conn.connected = False
        conn.connecting = False
        conn.incoming.clear()
        conn.outgoing.clear()  # Do NOT replay old commands after reconnecting.
        conn.next_retry = time.monotonic() + RECONNECT_INTERVAL_SEC
        if was_connected:
            logger.warning("%s: disconnected", conn.name)
            self.handle_disconnection(conn)

    def handle_disconnection(self, conn):
        """Define what an interrupted task means; do not silently resume it."""
        if conn is self.port1 and self.state == State.WAITING_FOR_ACK:
            # Task on Port 2 may still be executing; outcome is now uncertain.
            self.enter_fault("PORT_1_DISCONNECTED_DURING_TASK")
        # If idle, simply keep retrying. Port 1 requests will be rejected while
        # Port 2 is offline, rather than saved for later execution.

    # ------------------------------------------------------------------------
    # MESSAGE FRAMING AND TRANSPORT
    # ------------------------------------------------------------------------
    def send_json_port1(self, message):
        """Queue one JSON object followed by '\n' (TCP has no message boundaries)."""
        if not self.port1.connected:
            logger.warning("Cannot reply: Port 1 is disconnected")
            return False
        encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        self.port1.outgoing.extend(encoded)
        self.update_interest(self.port1)
        return True

    def flush_output(self, conn):
        """Send what the OS currently accepts; retain any unsent remainder."""
        try:
            count = conn.sock.send(conn.outgoing)
            if count == 0:
                self.disconnect(conn)
                return
            del conn.outgoing[:count]
            self.update_interest(conn)
        except BlockingIOError:
            pass  # Retry when the socket is writable again.
        except OSError:
            self.disconnect(conn)

    def read_available(self, conn):
        """Read without blocking, then extract all COMPLETE newline frames."""
        try:
            chunk = conn.sock.recv(4096) # extract the data
            if not chunk:
                self.disconnect(conn)  # EOF means connection was closed normally.
                return
            conn.incoming.extend(chunk)
            if len(conn.incoming) > MAX_MESSAGE_BYTES:
                logger.error("%s: input exceeds maximum size", conn.name)
                self.disconnect(conn)
                return
            # parse the message
            while b"\n" in conn.incoming:
                line, _, remainder = conn.incoming.partition(b"\n")
                conn.incoming = bytearray(remainder)
                if not line.strip():
                    continue
                if conn is self.port1:
                    try:
                        message = json.loads(line.decode("utf-8"))
                    except (ValueError, UnicodeError):
                        logger.warning("Port 1 sent invalid JSON")
                        continue
                    self.handle_port1_message(message)
                if not conn.connected:
                    break
        except BlockingIOError:
            pass
        except OSError:
            self.disconnect(conn)

    # [x,y,"color", angle (0-45)]
    # ------------------------------------------------------------------------
    # APPLICATION HANDLERS -- customize these for your project.
    # ------------------------------------------------------------------------
    def handle_port1_message(self, message):
        """Port 1 sends one request, then waits for a response."""
        if not isinstance(message, list) or len(message) != 4:
            logger.warning("Port 1 must send [x, y, color, angle]")
            return
        logger.info("Port 1 request: %s", message)
        # Check the CURRENT emergency flag, not past emergencies already cleared.

        if self.state == State.FAULT:
            self.send_json_port1({
                "status": "error",
                "error": "FAULT"
            })
            return
        if self.state != State.IDLE:
            self.send_json_port1({
                "status": "error",
                "error": "BUSY"
            })
            return

        #### ---- check if arm is still connected ---- ####
        self.ping_waldo()

        if not self.WALDO_CONNECTED: # make sure the server is running
            self.send_json_port1({
                "status": "error",
                "error": "ARM_DISCONNECTED"
            })
            return
        self.state = State.PROCESSING
        # try to process
        try:
            box = Box(message)
            if self.WALDO_CONNECTED:
                if not self.WALDO_PARK:
                    self.park(self.rbt)
                if self.move(box):
                    self.send_json_port1({"status": "SUCCESS",})
                else:
                    self.send_json_port1({"status": "FAILED",})
            else:
                logger.warning("Port 2 is disconnected")

        except Exception as e:
            logger.error("Processing failed: %s", e)

        finally:
            self.state = State.IDLE
            self.active_request_id = None

    def process_port1_request(self, message):
        """TODO: Your short (<1 second) local work goes here."""
        return message.get("data", {})

    # ------------------------------------------------------------------------
    # ARM MOVEMENT METHODS
    # ------------------------------------------------------------------------
    def move(self,box: Box):
        # check which area
        if self.LEFT_AREA.contains(box.x, box.y):
            return self.coordinate_left(box)
        elif self.CENTER_AREA.contains(box.x, box.y):
            # handle center
            return self.coordinate_center(box)

    def coordinate_zone(self, box:Box):
        # do something:
        # pickup angle:
        # x_min, y_min, x_max,y_max

        # make sure angle in range < 65
        self.ZONE_A = area(220,130,290,280)
        self.ZONE_B = area(226,-150,290,130)
        self.ZONE_C = area(200,-280,290,-150)
        self.ZONE_D = area(120,130,219,280)
        self.ZONE_E = area(0,0,0,0)
        self.ZONE_F = area(150,-100,200,0)
        self.ZONE_G = area(150,-280,200,-100)
        self.ZONE_G2 = area(120,-280,150,-150)
        self.ZONE_H = area(120,0,150,129)
        self.ZONE_I = area(120,-150,150,149)

        rx,rx,rz = 0,0,0

        if self.ZONE_B.contains(box.x,box.y):
            # Linear interpolation between two points
            def lerp(x, x1, y1, x2, y2):
                return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

            if not -45 <= box.angle <= 45:
                raise ValueError("Box angle must be between -45 and 45 degrees")

            angle = abs(box.angle)

            rx = lerp(angle, 0, 45, 45, 85)
            ry = lerp(angle, 0, -75, 45, -40)
            rz = lerp(angle, 0, 50, 45, 90)

            if box.angle < 0:
                rx = -rx
                rz = -rz
                # Optional: Round to integers if you don't want floating decimals
            round(rx), round(ry), round(rz)
        # if  Y = postive only check boxes A,D,E,H
        if box.y >= 0:
            rx = 75
            if self.ZONE_A.contains(box.x,box.y):               # --- CHECK ZONE A
                ry  = (box.angle)
                rz = 90
                if box.angle > 0:
                    ry = -box.angle - 45
            elif self.ZONE_D.contains(box.x,box.y):             # --- CHECK ZONE D
                    ry = round(box.angle)
                    rz = 90
            elif self.ZONE_E.contains(box.x,box.y):             # --- CHECK ZONE E
                print("cannot solve Zone E right now")
                return False
            elif self.ZONE_H.contains(box.x,box.y):             # --- CHECK ZONE H
                ry  = box.angle
                rz = 110
                if box.angle < 0:
                    ry = -box.angle + 45
            else:
                self.send_json_port1({
                    "status": "error",
                    "error": "Cannot Find Zone"
                })
        else: # check the other boxes
            rx = -75
            if self.ZONE_C.contains(box.x,box.y):                # --- CHECK ZONE C
                ry  = (box.angle)
                rz = -90
                if box.angle > 0:
                    ry = -box.angle - 45
            elif self.ZONE_F.contains(box.x,box.y):              # --- CHECK ZONE F
                if box.angle <= 30 or box.angle >= -30:
                    ry = round(box.angle)
                    rz = -90 if ry < 45 else -110
                else:
                    self.send_json_port1({
                    "status": "error",
                    "error": "Angle to big for zone F (can only move 30 to (-30))"
                })
            elif self.ZONE_G.contains(box.x,box.y):              # --- CHECK ZONE G
                ry = box.angle
                rz = -90 if ry < 45 else -110
            elif self.ZONE_G2.contains(box.x,box.y):             # --- CHECK ZONE G2
                ry = box.angle
                rz = -90 if ry < 45 else -110
            elif self.ZONE_I.contains(box.x,box.y):               # --- CHECK ZONE I
                ry  = box.angle
                if box.angle < 0:
                    ry = -box.angle + 45
                rz = -90 if rx < 45 else -110
            else:
                self.send_json_port1({
                    "status": "error",
                    "error": "Cannot Find Zone"
                })

    def (self, box:Box):
        """
        Calculate robot end-effector orientation (rx, ry, rz)
        based on the detected box angle (-45 to 45 degrees).
        """
        # Linear interpolation between two points
        def lerp(x, x1, y1, x2, y2):
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

        if not -45 <= box.angle <= 45:
            raise ValueError("Box angle must be between -45 and 45 degrees")

        angle = abs(box.angle)

        rx = lerp(angle, 0, 45, 45, 85)
        ry = lerp(angle, 0, -75, 45, -40)
        rz = lerp(angle, 0, 50, 45, 90)

        if box.angle < 0:
            rx = -rx
            rz = -rz
            # Optional: Round to integers if you don't want floating decimals
        round(rx), round(ry), round(rz)
        return self.pickup_at_point(box.x,box.y,rx,ry,rz)



    def pickup_at_point (self,x,y,rx,ry,rz):
        pose = [x,y,10,rx,ry,rz]
        # print(f"X: {x}  Y: {y}  Z: {10}  Rx: {rx}  Ry: {ry}  Rz: {rz}")
        try:
            self.WALDO_PARK = False
            # move to location
            # print(f"Moving to: {pose}")
            self.rbt.move_j(pose=pose, speed=MOVE_SPEED)
            # step down to item
            time.sleep(1)
            self.rbt.move_l([-PICK_UP_Z, 0, 0, 0, 0, 0], speed=0.25, frame="TRF", rel=True)
            time.sleep(1)
            self.rbt.move_l([PICK_UP_Z, 0, 0, 0, 0, 0], speed=0.25, frame="TRF", rel=True)
            time.sleep(2)
            # go back to park
            self.park()
            # print(f"Actual pose after move: {self.rbt.pose()}")
            return True
        except MotionError as e:
            print(f"Motion rejected: {e}")
            return False
        except Exception as e:
            print(f"Move failed: {e}")
            return False


    def set_emergency(self, reason):
        """Record current emergency; do NOT notify idle Port 1 immediately."""
        self.emergency_active = True
        self.emergency_reason = reason
        logger.warning("Emergency active: %s", reason)
        if self.state in (State.PROCESSING, State.WAITING_FOR_ACK):
            self.enter_fault("EMERGENCY_DURING_TASK")
            # An interrupted task cannot later be reported as successful.

    def clear_emergency(self):
        """Clear current status only; do NOT automatically clear FAULT state."""
        self.emergency_active = False
        self.emergency_reason = None
        logger.info("Emergency cleared")

    def enter_fault(self, reason):
        """Record failure. Do not send unsolicited emergency notifications."""
        self.current_fault = reason
        self.state = State.FAULT
        self.active_request_id = None
        logger.error("Coordinator entered FAULT: %s", reason)

    def handle_fault_recovery(self):
        """TODO: Decide recovery policy later. Conservative default: stay FAULT.

        You may later implement a manual reset, verified automatic recovery,
        or another policy. Never automatically replay an interrupted task.
        """
        return False

    def attempt_recovery(self):
        """A future RESET command or operator action can call this."""
        if (self.state == State.FAULT and not self.emergency_active
                and self.handle_fault_recovery()):
            self.current_fault = None
            self.active_request_id = None
            self.state = State.IDLE
            return True
        return False

    def check_arm(self):
        p = self.rbt.ping()
        if p is None or p.hardware_connected is None:
            print("Server not responding")
        elif not p.hardware_connected:
            print("Arm serial port DISCONNECTED")
        else:
            print("Arm connected")

    def connect_and_home_park(self) -> None:
        self.rbt.select_tool(TOOL)
        current = self.rbt.angles()
        #check if the robot is already close to home
        if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
            print("Homing...")
            self.rbt.home(wait=True)
        if current is None or max(abs(a - h) for a, h in zip(current, PARK_ANGLES)) > PARK_TOLERANCE_DEG:
            print("Parking...")
            try:
                self.rbt.move_j(pose=PARK_ANGLES, speed=MOVE_SPEED)
                self.WALDO_PARK = True
            except MotionError as e:
                    print(f"Cannot move to park: {e}")
                    self.WALDO_PARK = False

    def park(self) -> None:
        current = self.rbt.angles()
        if current is None or max(abs(a - h) for a, h in zip(current, PARK_ANGLES)) > PARK_TOLERANCE_DEG:
            print("Parking...")
            try:
                self.rbt.move_j(pose=PARK_ANGLES, speed=MOVE_SPEED)
                self.WALDO_PARK = True
            except MotionError as e:
                    print(f"Cannot move to park: {e}")
                    self.WALDO_PARK = False

    def ping_waldo(self):
        try:
            p = self.rbt.ping()
            if p is None or p.hardware_connected is None:
                print("ARM NOT responding")
                self.WALDO_CONNECTED = False
            else: # server is connected
                self.WALDO_CONNECTED = True
        except Exception as e:
            print("ERROR: Server NOT responding:", e)
            self.WALDO_CONNECTED = False

    # ------------------------------------------------------------------------
    # MAIN EVENT LOOP
    # ------------------------------------------------------------------------
    def run(self):
        """One thread monitors BOTH sockets; no busy waiting while connected.

        With two healthy connections and no queued output, select(None)
        waits indefinitely for activity. When offline, a finite timeout lets
        us retry connections and check unfinished connection attempts.
        """

        ''' create the areas '''
        logger.info("Coordinator starting")
        self.rbt = RobotClient(host=HOST, port=PORT)
        try:
            while self.running:
                # check if server is running
                # this should only run once
                self.ping_waldo()
                if not self.WALDO_CONNECTED:
                    self.connect_and_home_park()
                    self.ping_waldo()
                    print("Waldo Connected and homed")
                if not self.WALDO_PARK:
                    self.connect_and_home_park()

                # try to connect controller port
                conn = self.port1
                while self.running and self.WALDO_CONNECTED:
                    now = time.monotonic()
                    # get time for timeout: ONLY IF WE ARE STILL CONNECTING
                    if conn.connecting and now - conn.connect_started >= CONNECT_TIMEOUT_SEC:
                        self.disconnect(conn)
                    self.try_connect(conn)
                    timeout = None if conn.connected else RECONNECT_INTERVAL_SEC
                    '''
                        - Key: the Socket that sent a message:
                        - Mask: type of data sent
                            - EVENT_WRITE: buffer ready or connection was finished
                            - EVENT_READ: data ready to read or Connection was closed
                            _ else: we already established a connection and can immediately send data
                        this will block until we receive an update
                    '''
                    for key, mask in self.selector.select(timeout):
                        conn = key.data
                        # Ignore Port 2: shouldn't happen
                        if conn is not self.port1:
                            continue
                        # Check if Port 1 is still connecting
                        if conn.connecting:
                            if mask & selectors.EVENT_WRITE:
                                self.finish_connect(conn)
                            continue
                        # Check for incoming data or disconnection
                        if mask & selectors.EVENT_READ:
                            self.read_available(conn)
                        # Send any queued outgoing data
                        if conn.connected and mask & selectors.EVENT_WRITE:
                            self.flush_output(conn)
                print("for loop done:")

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.disconnect(self.port1)
            self.selector.close()
            logger.info("Coordinator stopped")


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    Coordinator().run()
