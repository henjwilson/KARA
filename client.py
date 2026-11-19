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
            chunk = conn.sock.recv(4096)
            if not chunk:
                self.disconnect(conn)  # EOF means peer closed normally.
                return
            conn.incoming.extend(chunk)
            if len(conn.incoming) > MAX_MESSAGE_BYTES:
                logger.error("%s: input exceeds maximum size", conn.name)
                self.disconnect(conn)
                return
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

    # ------------------------------------------------------------------------