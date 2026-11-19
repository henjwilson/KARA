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

import robot_defs as robot_defs
from parol6 import RobotClient
from parol6.utils.errors import MotionError

# ============================================================================
# GLOBAL CONFIGURATION -- change these when testing.
# These are the SERVER addresses/ports that this client connects TO.
# ============================================================================
HOST, PORT = "127.0.0.1", 5001
TOOL = "PNEUMATIC"

HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
HOME_TOLERANCE_DEG = 2.0

PARK_ANGLES = [0,250,45,90,0,90]
PARK_TOLERANCE_DEG = 2.0

MOVE_SPEED = 0.25


PORT1_HOST = "127.0.0.1"
PORT1_PORT = 5001
PORT2_HOST = "127.0.0.1"
PORT2_PORT = 5002

# When disconnected, retry without holding up the other connection.
RECONNECT_INTERVAL_SEC = 2.0
# How long an unfinished TCP connection attempt may remain pending.
CONNECT_TIMEOUT_SEC = 5.0
# Upper bound on a single incoming newline-delimited message.
MAX_MESSAGE_BYTES = 1024 * 1024
# For tests, logs show connections, received messages, and state changes.
LOG_LEVEL = logging.INFO

logger = logging.getLogger("coordinator")



class State(Enum):
    IDLE = auto()             # Ready to accept one new request from Port 1.
    PROCESSING = auto()       # Doing the short local processing step.
    WAITING_FOR_ACK = auto()  # Port 2 has the task; awaiting its completion.
    FAULT = auto()            # A fault occurred; recovery policy decides next step.


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
        self.port2 = Connection("PORT2", PORT2_HOST, PORT2_PORT)
        self.state = State.IDLE
        self.emergency_active = False
        self.emergency_reason = None
        self.current_fault = None
        self.active_request_id = None
        self.running = True

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
            logger.warning("%s: connect failed: %s", conn.name, exc)
            self.disconnect(conn)

    def finish_connect(self, conn):
        """Verify that a writable connecting socket actually connected."""
        error = conn.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if error:
            logger.warning("%s: connection refused/failed (code=%s)", conn.name, error)
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
        """Close one connection; leave the OTHER connection running."""
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
        if conn is self.port2 and self.state == State.WAITING_FOR_ACK:
            self.enter_fault("PORT_2_DISCONNECTED_DURING_TASK")
        elif conn is self.port1 and self.state == State.WAITING_FOR_ACK:
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

    def queue_port2_bytes(self, payload):
        """Port 2's wire format is TBD. This function only sends raw bytes."""
        if not self.port2.connected:
            return False
        self.port2.outgoing.extend(payload)
        self.update_interest(self.port2)
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
                self.disconnect(conn)  # EOF means peer closed normally.
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
                else:
                    self.handle_port2_raw_message(bytes(line))
                if not conn.connected:
                    break
        except BlockingIOError:
            pass
        except OSError:
            self.disconnect(conn)

    # ------------------------------------------------------------------------
    # APPLICATION HANDLERS -- customize these for your project.
    # ------------------------------------------------------------------------
    def handle_port1_message(self, message):
        """Port 1 sends one request, then waits for a response."""
        if not isinstance(message, dict):
            logger.warning("Port 1 JSON must be an object")
            return
        request_id = message.get("request_id")
        logger.info("Port 1 request: %s", message)

        # Check the CURRENT emergency flag, not past emergencies already cleared.
        if self.emergency_active:
            self.reply_error(request_id, "EMERGENCY_ACTIVE", self.emergency_reason)
            return
        if self.state == State.FAULT:
            self.reply_error(request_id, "COORDINATOR_FAULT", self.current_fault)
            return
        if self.state != State.IDLE:
            self.reply_error(request_id, "BUSY")  # No request queue.
            return
        if not self.port2.connected:
            self.reply_error(request_id, "PORT_2_UNAVAILABLE")
            return

        self.active_request_id = request_id
        self.state = State.PROCESSING
        try:
            result = self.process_port1_request(message)
            # Check again: process_port1_request is expected to be quick, but
            # cannot be interrupted by selector events while it is running.
            if self.emergency_active or not self.port2.connected:
                raise RuntimeError("Port 2 unavailable or emergency active")
            sent = self.send_task_to_port2(result)
            if not sent:
                raise RuntimeError("Port 2 task send failed")
            self.state = State.WAITING_FOR_ACK
        except Exception as exc:
            logger.exception("Request processing/dispatch failed")
            self.reply_error(request_id, "DISPATCH_FAILED", str(exc))
            self.active_request_id = None
            self.state = State.IDLE

    def process_port1_request(self, message):
        """TODO: Your short (<1 second) local work goes here."""
        return message.get("data", {})

    def send_task_to_port2(self, processed_data):
        """TODO: Translate processed data into Port 2's eventual protocol.

        Return True ONLY after a task has been queued for sending.
        Currently returns False deliberately: no Port 2 format is assumed.
        For example, later you might call queue_port2_bytes(payload).
        """
        logger.info("Port 2 task adapter not implemented: %r", processed_data)
        return False

    def handle_port2_raw_message(self, raw_message):
        """TODO: Decode Port 2 messages and call the appropriate handler.

        Expected semantic events (wire format intentionally unspecified):
          - emergency reported -> set_emergency(reason)
          - emergency cleared  -> clear_emergency()
          - task completed     -> handle_port2_completion(success=True)
          - task failed        -> handle_port2_completion(success=False)

        Until you implement this adapter, Port 2 messages are only logged.
        The shell temporarily assumes newline-delimited Port 2 frames too;
        replace read_available() framing for Port 2 if needed.
        """
        logger.info("Port 2 raw message (adapter TBD): %r", raw_message)

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

    def handle_port2_completion(self, success, reason=None):
        """Called by the future Port 2 decoder when the active task finishes."""
        if self.state != State.WAITING_FOR_ACK:
            logger.warning("Ignoring completion with no active waiting task")
            return
        request_id = self.active_request_id
        self.active_request_id = None
        if success and not self.emergency_active:
            self.send_json_port1({"type": "response", "request_id": request_id,
                                  "status": "success"})
            self.state = State.IDLE
        else:
            self.enter_fault(reason or "PORT_2_TASK_FAILED")

    def reply_error(self, request_id, code, reason=None):
        response = {"type": "response", "request_id": request_id,
                    "status": "error", "error": code}
        if reason is not None:
            response["reason"] = reason
        self.send_json_port1(response)

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

    def check_arm(rbt):
        p = rbt.ping()
        if p is None or p.hardware_connected is None:
            print("Server not responding")
        elif not p.hardware_connected:
            print("Arm serial port DISCONNECTED")
        else:
            print("Arm connected")

    def connect_and_home(rbt: RobotClient) -> None:
        rbt.select_tool(TOOL)
        current = rbt.angles()
        #check if the robot is already close to home
        if current is None or max(abs(a - h) for a, h in zip(current, HOME_ANGLES)) > HOME_TOLERANCE_DEG:
            print("Homing...")
            rbt.home(wait=True)
        if current is None or max(abs(a - h) for a, h in zip(current, PARK_ANGLES)) > PARK_TOLERANCE_DEG:
            print("Parking...")
            try:
                rbt.move_j(pose=PARK_ANGLES, speed=MOVE_SPEED)
            except MotionError as e:
                    print(f"Cannot move to park: {e}")

    # ------------------------------------------------------------------------
    # MAIN EVENT LOOP
    # ------------------------------------------------------------------------
    def run(self):
        """One thread monitors BOTH sockets; no busy waiting while connected.

        With two healthy connections and no queued output, select(None)
        waits indefinitely for activity. When offline, a finite timeout lets
        us retry connections and check unfinished connection attempts.
        """
        logger.info("Coordinator starting")
        rbt = RobotClient(host=HOST, port=PORT)

        try:
            while self.running:
                # get current time for logger
                now = time.monotonic()

                try:
                    p = rbt.ping()
                    if p is None or p.hardware_connected is None:
                        print("Server NOT responding")
                    else: # server is connected
                        connect_and_home(rbt)
                except Exception as e:
                    print("Server NOT responding:", e)

                # try to connect to ports
                for conn in (self.port1, self.port2):
                    if conn.connecting and now - conn.connect_started >= CONNECT_TIMEOUT_SEC:
                        logger.warning("%s: connection attempt timed out", conn.name)
                        self.disconnect(conn)
                    self.try_connect(conn)

                ''' check is a port is not connected: if so we need to wait for both ports to be connected'''
                waiting_for_connection = any(
                    # if any port is not connected this returns true
                    not conn.connected for conn in (self.port1, self.port2)
                )

                # Missing Connection: program wakes up every .25 second and retry connection
                # Both Connected: stay awake indefinitely waiting for message
                timeout = 0.25 if waiting_for_connection else None

                '''
                    - Key: the Socket that sent a message:
                    - Mask: type of data sent
                        - EVENT_WRITE: buffer ready or connection was finished
                        - EVENT_READ: data ready to read or Connection was closed
                        _ else: we already established a connection and can immediately send data
                    this will block until we receive an update
                '''
                for key, mask in self.selector.select(timeout): # wait for socket status changed
                    conn = key.data # get the port
                    if conn.connecting: # if we are still trying to connect to that port we might received a notification that socket is ready to use
                        if mask & selectors.EVENT_WRITE: # check the event: read,write, ect
                            self.finish_connect(conn) # update status that we can read//write
                        continue
                    if mask & selectors.EVENT_READ: # Either we good data or Connection was closed
                        self.read_available(conn)
                    if conn.connected and mask & selectors.EVENT_WRITE: # we can read or write
                        self.flush_output(conn)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            for conn in (self.port1, self.port2):
                self.disconnect(conn)
            self.selector.close()
            logger.info("Coordinator stopped")


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    Coordinator().run()
