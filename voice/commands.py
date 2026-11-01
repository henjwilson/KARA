"""Strict voice-command grammar. No subprocesses or device access."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    kind: str
    key: str = ""


# Speech-to-text may spell the name Kara as Cara or "car a".
WAKE_PATTERN = re.compile(r"^(?:hey|hi|ok|okay) (?:kara|cara|car a)(?: |$)")


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return " ".join(text.split())


def strip_wake(text: str) -> tuple[bool, str]:
    text = normalize(text)
    match = WAKE_PATTERN.match(text)
    if match:
        return True, text[match.end():].strip()
    return False, text


# Only explicitly listed utterances can become commands.
PROGRAM_PHRASES = {
    "seek yellow": "seek_yellow",
    "seek the yellow": "seek_yellow",
    "find yellow": "seek_yellow",
    "pick up yellow": "seek_yellow",
    "pick up the yellow block": "seek_yellow",
    "seek yellow dry run": "seek_yellow_dry",
    "dry run seek yellow": "seek_yellow_dry",
    "test seek yellow": "seek_yellow_dry",
    "track yellow": "track_blocks",
    "track blocks": "track_blocks",
}
ACTION_PHRASES = {
    "home": "home", "go home": "home",
    "park": "park", "go to park": "park",
    "open gripper": "gripper_open", "close gripper": "gripper_close",
    "read pose": "pose", "get pose": "pose",
}
STOP_PHRASES = {"stop", "stop program", "stop robot", "halt", "halt robot"}


def parse_command(text: str) -> Intent | None:
    phrase = normalize(text)
    if phrase in STOP_PHRASES:
        return Intent("stop")
    if phrase in ("help", "commands", "what can you do"):
        return Intent("help")
    if phrase in ("status", "robot status", "what is running"):
        return Intent("status")
    if phrase == "cancel":
        return Intent("cancel")
    if phrase.startswith("confirm "):
        inner = parse_command(phrase.removeprefix("confirm "))
        if inner and inner.kind in ("program", "action"):
            return Intent("confirm", inner.key)
        return None
    if phrase in PROGRAM_PHRASES:
        return Intent("program", PROGRAM_PHRASES[phrase])
    if phrase in ACTION_PHRASES:
        return Intent("action", ACTION_PHRASES[phrase])
    return None
