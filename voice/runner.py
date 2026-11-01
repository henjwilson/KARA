"""Launch only existing, allowlisted KARA scripts; no shell execution."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from programs.kara_launcher_core import (
    PROGRAM_BY_KEY,
    action_command,
    locate_python,
    program_command,
)


class ProgramRunner:
    def __init__(self, *, python: str | None = None, camera: int = 0, preview: bool = False):
        self.python = python or locate_python()
        self.camera = camera
        self.preview = preview
        self.process: subprocess.Popen | None = None
        self.active_name: str | None = None

    def running(self) -> bool:
        if self.process is not None and self.process.poll() is not None:
            print(f"[KARA] {self.active_name} finished (exit {self.process.returncode}).", flush=True)
            self.process = None
            self.active_name = None
        return self.process is not None

    def start_program(self, key: str) -> bool:
        if key not in PROGRAM_BY_KEY:
            raise ValueError("Unrecognized program")
        return self._start(program_command(key, self.python, camera=self.camera), key)

    def start_action(self, action: str) -> bool:
        if action not in {"home", "park", "gripper_open", "gripper_close", "pose"}:
            raise ValueError("Unrecognized action")
        return self._start(action_command(action, self.python), action)

    def _start(self, command: list[str], name: str) -> bool:
        if self.running():
            print(f"[KARA] Busy running {self.active_name}. Say 'Hey Kara, stop' first.", flush=True)
            return False
        print(f"[KARA] {'WOULD RUN' if self.preview else 'Starting'} {name}: {command!r}", flush=True)
        if self.preview:
            return True
        # Scripts inherit stdout/stderr so that their OpenCV and robot logs remain visible.
        kwargs: dict = {"cwd": str(Path(__file__).resolve().parents[1])}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(command, **kwargs)
        self.active_name = name
        return True

    def stop(self) -> None:
        # Request backend robot STOP *before* terminating script execution.
        # This is a SOFTWARE stop, NOT a physical emergency stop.
        print("[KARA] Requesting software stop (NOT a hardware E-STOP).", flush=True)
        if not self.preview:
            try:
                result = subprocess.run(
                    action_command("stop", self.python),
                    cwd=str(Path(__file__).resolve().parents[1]),
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode:
                    print(f"[KARA] Backend stop failed: {result.stderr.strip()}", flush=True)
                else:
                    print("[KARA] Backend software-stop request sent.", flush=True)
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"[KARA] Could not complete backend stop request: {exc}", flush=True)
        if not self.running():
            return
        if self.preview:
            return
        assert self.process is not None
        try:
            self.process.terminate()
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("[KARA] Program did not exit; forcing termination.", flush=True)
            self.process.kill()
            self.process.wait(timeout=3)
        except OSError as exc:
            print(f"[KARA] Could not terminate program: {exc}", flush=True)
        finally:
            self.running()

    def close(self) -> None:
        if self.running():
            print("[KARA] Voice listener ending; stopping launched program.", flush=True)
            self.stop()
