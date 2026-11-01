"""Wake phrase / confirmation state machine; dependency-free and unit-testable."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from programs.kara_launcher_core import PROGRAM_BY_KEY
from voice.commands import Intent, parse_command, strip_wake
from voice.runner import ProgramRunner


MOTION_ACTIONS = {"home", "park", "gripper_open", "gripper_close"}


@dataclass
class Pending:
    kind: str
    key: str
    until: float


class KaraAssistant:
    def __init__(self, runner: ProgramRunner, *, enable_motion: bool = False,
                 wake_seconds: float = 6.0, confirm_seconds: float = 8.0,
                 clock: Callable[[], float] = time.monotonic):
        self.runner = runner
        self.enable_motion = enable_motion
        self.wake_seconds = wake_seconds
        self.confirm_seconds = confirm_seconds
        self.clock = clock
        self.wake_until = 0.0
        self.pending: Pending | None = None

    def handle(self, utterance: str) -> str:
        now = self.clock()
        if self.pending and now > self.pending.until:
            self.pending = None
        has_wake, phrase = strip_wake(utterance)
        if has_wake and not phrase:
            self.wake_until = now + self.wake_seconds
            return "Listening. Say a command within six seconds."

        # STOP is accepted with the wake phrase OR as a standalone instruction,
        # reducing delays when an operation needs to be interrupted.
        intent = parse_command(phrase)
        if intent and intent.kind == "stop":
            self.pending = None
            self.wake_until = 0.0
            self.runner.stop()
            return "Software stop requested. Use the hardware E-STOP for an emergency."

        # Confirmation is allowed without wake phrase only while a command is armed.
        if intent and intent.kind == "confirm" and self.pending:
            pending = self.pending
            if now <= pending.until and intent.key == pending.key:
                self.pending = None
                self.wake_until = 0.0
                return self._launch(pending.kind, pending.key)
            return f"Confirmation did not match. Say 'confirm {pending.key.replace('_', ' ')}'."

        if not has_wake and now > self.wake_until:
            return ""  # Ignore background speech.
        if not intent:
            return "Command not recognized; say 'Hey Kara, help' for examples."
        self.wake_until = 0.0
        if intent.kind == "cancel":
            self.pending = None
            return "Pending command canceled."
        if intent.kind == "help":
            return ("Try: Hey Kara, seek yellow; seek yellow dry run; track yellow; "
                    "home; open gripper; status; or stop. Motion requires confirmation.")
        if intent.kind == "status":
            name = self.runner.active_name if self.runner.running() else "nothing"
            mode = "motion enabled" if self.enable_motion else "motion disabled"
            return f"Running: {name}. {mode}."
        if intent.kind == "confirm":
            return "No action is waiting for confirmation."
        if intent.kind in ("program", "action"):
            return self._request(intent, now)
        return "Command not recognized."

    def _request(self, intent: Intent, now: float) -> str:
        if intent.kind == "program":
            spec = PROGRAM_BY_KEY[intent.key]
            if spec.moves_robot:
                if not self.enable_motion:
                    if intent.key == "seek_yellow":
                        if self.runner.start_program("seek_yellow_dry"):
                            return "Robot motion disabled. Starting yellow detection DRY RUN."
                        return "Another program is running."
                    return "Motion disabled. Restart with --enable-motion to run this routine."
                self.pending = Pending("program", intent.key, now + self.confirm_seconds)
                return f"Motion requested: {spec.title}. Say 'confirm seek yellow' within eight seconds."
            if self.runner.start_program(intent.key):
                return f"Started {spec.title}."
            return "Another program is running."
        if intent.key in MOTION_ACTIONS:
            if not self.enable_motion:
                return "Motion disabled. Restart with --enable-motion to control the robot."
            self.pending = Pending("action", intent.key, now + self.confirm_seconds)
            return f"Robot action: {intent.key.replace('_', ' ')}. Say 'confirm {intent.key.replace('_', ' ')}' within eight seconds."
        return self._launch(intent.kind, intent.key)

    def _launch(self, kind: str, key: str) -> str:
        if kind == "program":
            okay = self.runner.start_program(key)
        elif kind == "action":
            okay = self.runner.start_action(key)
        else:
            return "Unknown action."
        return f"Starting {key.replace('_', ' ')}." if okay else "Another program is running."
