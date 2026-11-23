"""Wake phrase / confirmation state machine; dependency-free and unit-testable."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from programs.kara_launcher_core import PROGRAM_BY_KEY
from voice.commands import ACTION_PHRASES, PROGRAM_PHRASES, Intent, parse_command, strip_wake
from voice.runner import ProgramRunner


MOTION_ACTIONS = {"home", "park", "gripper_open", "gripper_close"}


def say_name(key: str) -> str:
    # the words you actually say for a command, like "open gripper" not "gripper open"
    key = key.removeprefix("action:")
    for phrase, phrase_key in list(ACTION_PHRASES.items()) + list(PROGRAM_PHRASES.items()):
        if phrase_key == key:
            return phrase
    return key.replace("_", " ")


@dataclass
class Pending:
    kind: str
    key: str
    until: float


class KaraAssistant:
    def __init__(self, runner: ProgramRunner, *, enable_motion: bool = False,
                 wake_seconds: float = 6.0, confirm_seconds: float = 8.0,
                 clock: Callable[[], float] = time.monotonic,
                 motion_hint: str = "Restart with --enable-motion",
                 require_wake: bool = True):
        # motion_hint is what she says when motion is off, the gui and old script fix it differently
        # require_wake=False is for the gui, no "hey kara" needed
        self.runner = runner
        self.enable_motion = enable_motion
        self.wake_seconds = wake_seconds
        self.confirm_seconds = confirm_seconds
        self.clock = clock
        self.motion_hint = motion_hint
        self.require_wake = require_wake
        self.wake_until = 0.0
        self.pending: Pending | None = None

    def handle(self, utterance: str) -> str:
        now = self.clock()
        if self.pending and now > self.pending.until:
            self.pending = None
        has_wake, phrase = strip_wake(utterance)
        # no wake word means every phrase counts as awake
        if not self.require_wake:
            has_wake = True
        if has_wake and not phrase:
            self.wake_until = now + self.wake_seconds
            return "I'm listening. Say a command."

        # STOP is accepted with the wake phrase OR as a standalone instruction,
        # reducing delays when an operation needs to be interrupted.
        intent = parse_command(phrase)
        if intent and intent.kind == "stop":
            self.pending = None
            self.wake_until = 0.0
            self.runner.stop()
            return "Software stop requested. Use the hardware E-stop in an emergency."

        # Confirmation is allowed without wake phrase only while a command is armed.
        if intent and intent.kind == "confirm" and self.pending:
            pending = self.pending
            if now <= pending.until and intent.key == pending.key:
                self.pending = None
                self.wake_until = 0.0
                return self._launch(pending.kind, pending.key)
            # tell them the exact words to say
            return f"That did not match. Say confirm {say_name(pending.key)}."

        if not has_wake and now > self.wake_until:
            return ""  # Ignore background speech.
        if not intent:
            if not self.require_wake:
                return ""  # no wake word, so stay quiet on random talk
            return "Command not recognized; say 'Hey Kara, help' for examples."
        self.wake_until = 0.0
        if intent.kind == "cancel":
            self.pending = None
            return "Okay, canceled."
        if intent.kind == "help":
            # one short list, the old one had semicolons the voice just ran together
            wake = "Hey Kara, " if self.require_wake else ""
            return (f"You can say {wake}seek yellow, track yellow, home, park, open gripper, "
                    "close gripper, status, or stop. Anything that moves the robot needs a confirm.")
        if intent.kind == "status":
            # full sentences, the voice reads flat with no real pauses so wording matters
            # old reply was "running: nothing. motion disabled." and sounded like one word
            running = (f"{say_name(self.runner.active_name)} is running." if self.runner.running()
                       else "Nothing is running.")
            mode = "Motion is enabled." if self.enable_motion else "Motion is currently disabled."
            return f"{running} {mode}"
        if intent.kind == "confirm":
            return "Nothing is waiting for a confirm."
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
                            return "Robot motion is off, so I'm starting the yellow detection DRY RUN."
                        return "Another program is running."
                    return f"Motion disabled. {self.motion_hint} to run this routine."
                self.pending = Pending("program", intent.key, now + self.confirm_seconds)
                return self._ask_confirm(intent.key)
            if self.runner.start_program(intent.key):
                return f"Starting {say_name(intent.key)}."
            return "Another program is running."
        if intent.key in MOTION_ACTIONS:
            if not self.enable_motion:
                return f"Motion disabled. {self.motion_hint} to control the robot."
            self.pending = Pending("action", intent.key, now + self.confirm_seconds)
            return self._ask_confirm(intent.key)
        return self._launch(intent.kind, intent.key)

    def _launch(self, kind: str, key: str) -> str:
        if kind == "program":
            okay = self.runner.start_program(key)
        elif kind == "action":
            okay = self.runner.start_action(key)
        else:
            return "Unknown action."
        return f"Starting {say_name(key)}." if okay else "Another program is running."

    def _ask_confirm(self, key: str) -> str:
        # kept short, the long version used up half the 8 second confirm window
        name = say_name(key)
        return f"Please say confirm {name}."