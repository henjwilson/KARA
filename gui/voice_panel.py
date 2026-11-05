# voice commands for the kara gui
# the mic runs in a background thread, and commands go through the gui's own
# button code so voice gets the same safety checks

from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from voice.assistant import KaraAssistant, say_name
from voice.commands import ACTION_PHRASES, PROGRAM_PHRASES, STOP_PHRASES


MODEL_PATH = Path(__file__).resolve().parents[1] / 'voice' / 'models' / 'vosk-model-small-en-us-0.15'
CHECK_ICON = (Path(__file__).resolve().parent / 'check.svg').as_posix()

SPOKEN_ERRORS = {
    'backend not reachable': 'The robot server is not running.',
    'motion is locked': 'Robot motion is locked.',
    'already running': 'Another program is already running.',
    'stop in progress': 'A stop is still in progress.',
}


def _label(text: str, obj: str, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setObjectName(obj)
    label.setWordWrap(wrap)
    return label


def list_mics() -> list[tuple[int, str]]:
    try:
        import sounddevice as sd
        return [(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
    except (ImportError, OSError):
        return []


def speak_command(text: str) -> tuple[list[str], dict[str, str]] | None:
    # built in voices
    if sys.platform == 'darwin':
        return ['say', text], {}

    if os.name == 'nt':
        script = ('Add-Type -AssemblyName System.Speech; '
                  '(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:KARA_SAY)')
        return ['powershell', '-NoProfile', '-Command', script], {'KARA_SAY': text}

    if shutil.which('espeak'):
        return ['espeak', text], {}

    return None


def command_phrases() -> list[str]:
    # only these phrases can be heard
    commands = list(PROGRAM_PHRASES) + list(ACTION_PHRASES)
    extras = ['help', 'commands', 'what can you do', 'status', 'robot status', 'what is running', 'cancel']

    return commands + list(STOP_PHRASES) + extras + [f'confirm {c}' for c in commands]


def spoken_error(error: str) -> str:
    for match, sentence in SPOKEN_ERRORS.items():
        if match in error.lower():
            return f"Sorry, I can't do that. {sentence}"

    return "Sorry, I can't do that. Check the activity log."


class VoiceWorker(QThread):
    heard = Signal(str)
    ready = Signal(int)
    failed = Signal(str)

    def __init__(self, device: int | None, parent: QWidget | None = None):
        super().__init__(parent)
        self.device = device
        self._stop = threading.Event()
        self._paused = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def run(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError:
            self.failed.emit('Voice packages missing. Run: python -m pip install -r voice/requirements.txt')
            return

        if not (MODEL_PATH / 'conf').is_dir() or not (MODEL_PATH / 'am').is_dir():
            self.failed.emit('Speech model missing. Run: python -m voice.setup_model')
            return

        try:
            SetLogLevel(-1)
            model = Model(str(MODEL_PATH))

            # use the mic's own rate, vosk converts it
            rate = int(sd.query_devices(self.device, 'input')['default_samplerate'])
            rec = KaldiRecognizer(model, rate, json.dumps(command_phrases() + ['[unk]']))
            samples: queue.Queue[bytes] = queue.Queue(maxsize=20)

            def on_audio(indata, frames, time_info, status):
                try:
                    samples.put_nowait(bytes(indata))
                except queue.Full:
                    pass

            with sd.RawInputStream(samplerate=rate, blocksize=4000, device=self.device,
                                   dtype='int16', channels=1, callback=on_audio):
                self.ready.emit(rate)
                was_paused = False

                while not self._stop.is_set():
                    try:
                        data = samples.get(timeout=0.3)
                    except queue.Empty:
                        continue

                    # mic is off while kara talks, so throw that audio away
                    if self._paused.is_set():
                        was_paused = True
                        continue

                    # start fresh so none of her voice leaks into the next command
                    if was_paused:
                        rec.Reset()
                        was_paused = False

                    if rec.AcceptWaveform(data):
                        # [unk] is anything that isn't a command
                        words = json.loads(rec.Result()).get('text', '').split()
                        text = ' '.join(w for w in words if w != '[unk]')
                        if text:
                            self.heard.emit(text)

        except (sd.PortAudioError, ValueError, OSError, RuntimeError) as exc:
            self.failed.emit(f'Microphone error: {exc}')


class DashboardRunner:
    # same methods as ProgramRunner in voice/runner.py, but it uses the gui

    def __init__(self, gui):
        self.gui = gui
        self.last_error = ''

    @property
    def active_name(self) -> str | None:
        return self.gui.current_program_key

    def running(self) -> bool:
        return self.gui._busy(self.gui.program_process)

    def start_program(self, key: str) -> bool:
        return self._start(self.gui.start_program, key)

    def start_action(self, action: str) -> bool:
        return self._start(self.gui.start_action, action)

    def _start(self, launch: Callable[..., bool], key: str) -> bool:
        try:
            return launch(key, voice=True)
        except (OSError, RuntimeError, ValueError) as exc:
            self.last_error = str(exc).splitlines()[0]
            return False

    def stop(self) -> None:
        self.gui.stop_robot()


class VoicePanel(QFrame):
    message = Signal(str, str)

    def __init__(self, runner: DashboardRunner, motion_enabled: Callable[[], bool],
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName('sideInfo')

        self.runner = runner
        self.motion_enabled = motion_enabled
        self.assistant = KaraAssistant(runner, motion_hint='Turn on robot motion',
                                       require_wake=False)
        self.worker: VoiceWorker | None = None
        self.model_loaded = False
        self.speaker = QProcess(self)
        self.speaker.finished.connect(self._done_speaking)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 13, 12, 14)
        lay.setSpacing(8)

        lay.addWidget(_label('●   VOICE COMMANDS', 'hint'))
        self.state_label = _label('Off', 'mutedText', wrap=True)
        lay.addWidget(self.state_label)

        self.mic_combo = QComboBox()
        self.mic_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mic_combo.setMinimumContentsLength(10)
        self.mic_combo.addItem('Default microphone', None)
        for index, name in list_mics():
            self.mic_combo.addItem(f'{index}: {name}', index)
        lay.addWidget(self.mic_combo)

        self.listen_btn = QPushButton('Start listening')
        self.listen_btn.setObjectName('softButton')
        self.listen_btn.setCursor(Qt.PointingHandCursor)
        self.listen_btn.clicked.connect(self.toggle)
        lay.addWidget(self.listen_btn)

        can_speak = speak_command('') is not None
        self.speak_box = QCheckBox('Spoken replies')
        self.speak_box.setChecked(can_speak)
        self.speak_box.setEnabled(can_speak)
        self.speak_box.setStyleSheet(
            'QCheckBox::indicator:checked { background: #46D5C3; border: 1px solid #55F0D6; '
            f'border-radius: 6px; image: url("{CHECK_ICON}"); }}'
        )
        lay.addWidget(self.speak_box)

        self.heard_label = _label('Heard: -', 'mutedText', wrap=True)
        self.reply_label = _label('Say "help" for commands.', 'mutedText', wrap=True)
        lay.addWidget(self.heard_label)
        lay.addWidget(self.reply_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(400)

    def listening(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def toggle(self) -> None:
        if self.worker is not None:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self) -> None:
        self.worker = VoiceWorker(self.mic_combo.currentData(), self)
        self.worker.heard.connect(self._on_heard)
        self.worker.ready.connect(self._on_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.model_loaded = False
        self.worker.start()

        self.listen_btn.setText('Stop listening')
        self.mic_combo.setEnabled(False)
        self.state_label.setText('Loading speech model...')

    def stop_listening(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.wait(2000)

        self.assistant.pending = None
        self.assistant.wake_until = 0.0

    def shutdown(self) -> None:
        self.timer.stop()
        self.stop_listening()
        self._stop_speaking()

    def speak(self, text: str) -> None:
        command = speak_command(text)
        if not self.speak_box.isChecked() or command is None:
            return

        argv, extra_env = command
        self._stop_speaking()

        env = QProcessEnvironment.systemEnvironment()
        for key, value in extra_env.items():
            env.insert(key, value)
        self.speaker.setProcessEnvironment(env)

        # mic off while she talks so she doesn't hear herself
        if self.worker is not None:
            self.worker.pause()

        self.speaker.start(argv[0], argv[1:])
        if not self.speaker.waitForStarted(1000):
            self.message.emit(f'Spoken reply failed: {self.speaker.errorString()}', 'error')
            self._done_speaking()

    def speaking(self) -> bool:
        return self.speaker.state() != QProcess.NotRunning

    def _done_speaking(self) -> None:
        # small gap so the room echo dies down before the mic comes back
        QTimer.singleShot(300, self._mic_back_on)

        # the confirm timer starts after she's done talking, not while she talks
        pending = self.assistant.pending
        if pending is not None:
            pending.until = time.monotonic() + self.assistant.confirm_seconds

    def _mic_back_on(self) -> None:
        if self.worker is not None and not self.speaking():
            self.worker.resume()

    def _stop_speaking(self) -> None:
        if self.speaking():
            self.speaker.finished.disconnect(self._done_speaking)
            self.speaker.kill()
            self.speaker.waitForFinished(500)
            self.speaker.finished.connect(self._done_speaking)

    def _on_ready(self, rate: int) -> None:
        self.model_loaded = True
        self.message.emit(f'Voice listening at {rate} Hz. Say "help".', 'system')

    def _on_failed(self, error: str) -> None:
        self.reply_label.setText(error)
        self.message.emit(error, 'error')

    def _on_finished(self) -> None:
        self.worker = None
        self.listen_btn.setText('Start listening')
        self.mic_combo.setEnabled(True)
        self._refresh()

    def _on_heard(self, text: str) -> None:
        self.heard_label.setText(f'Heard: "{text}"')
        self.assistant.enable_motion = self.motion_enabled()
        self.runner.last_error = ''

        reply = self.assistant.handle(text)
        spoken = reply
        if self.runner.last_error:
            reply = f'Could not run: {self.runner.last_error}'
            spoken = spoken_error(self.runner.last_error)
        if not reply:
            return

        self.reply_label.setText(reply)
        self.message.emit(f'Voice heard "{text}"', 'system')
        self.message.emit(f'Voice: {reply}', 'error' if self.runner.last_error else 'system')
        self.speak(spoken)
        self._refresh()

    def _refresh(self) -> None:
        now = time.monotonic()
        pending = self.assistant.pending

        if not self.listening():
            state = 'Off'
        elif not self.model_loaded:
            state = 'Loading speech model...'
        elif self.speaking():
            state = 'Talking, mic is off'
        elif pending and now <= pending.until:
            state = f'Waiting for "confirm {say_name(pending.key)}"'
        elif now <= self.assistant.wake_until:
            state = 'Awake, say a command'
        else:
            state = 'Listening for a command'

        self.state_label.setText(state)
