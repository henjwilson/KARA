# KARA Voice — standalone offline voice commands

A separate voice listener for your PAROL6 project. **Not connected to the Qt GUI.**
It reuses `programs/kara_launcher_core.py` and `programs/kara_robot_action.py`;
there are **no edits to the original robot scripts**.

## 1. Set up (Windows, Python 3.10 recommended)

From the **repository root** (the folder containing `programs/` and `OpenCV/`):

```powershell
# Use the same environment that already runs PAROL6 and OpenCV.
.\.venv\Scripts\python.exe -m pip install -r voice\requirements.txt
.\.venv\Scripts\python.exe -m voice.setup_model
```

If your robot environment is not `.venv`, replace the first part with the path to
its Python executable, e.g. `py -3.10`. Vosk and sounddevice are the only additional
voice dependencies. **Do not use Python 3.14 unless compatible Vosk wheels are available.**

The model download is **one-time**, about 40 MB, from the official [Vosk model
catalog](https://alphacephei.com/vosk/models). The file is SHA-256 verified and
installed in `voice/models/`, which is ignored by Git. Speech is recognized locally
after download; no online speech service is called.

## 2. Run

```powershell
# From repository root — safe default, yellow detection only (no robot motion)
.\.venv\Scripts\python.exe -m voice

# List available microphones (if the default mic is wrong)
.\.venv\Scripts\python.exe -m voice --list-devices

# Specify microphone and USB camera indices
.\.venv\Scripts\python.exe -m voice --device 1 --camera 0

# When voice and robot scripts are in different Python environments:
python -m voice --python "C:\\path\\to\\robot_venv\\Scripts\\python.exe"

# Actual robot movement is an EXPLICIT opt-in; spoken confirmation is mandatory.
.\.venv\Scripts\python.exe -m voice --enable-motion
```

Or double-click `voice/start_voice.bat` (safe mode) or
`voice/start_voice_real_motion.bat` (asks permission before enabling motion).

For best results use a headset mic and speak in a quiet room.

## 3. Voice commands

| Say | What happens |
|---|---|
| **Hey Kara, seek yellow** | Safe mode: starts `OpenCV/seek_yellow.py --dry-run`. Real-motion mode: asks you to **confirm seek yellow** first. |
| **Hey Kara, seek yellow dry run** | Starts the OpenCV dry run, regardless of motion mode. |
| **Hey Kara, track yellow** | Starts `OpenCV/main.py` to show the tracked camera image. |
| **Hey Kara, home** | Motion mode only; requires **confirm home**. |
| **Hey Kara, park** | Motion mode only; requires **confirm park**. |
| **Hey Kara, open gripper** | Motion mode only; requires **confirm open gripper**. |
| **Hey Kara, close gripper** | Motion mode only; requires **confirm close gripper**. |
| **Hey Kara, read pose** | Requests joint angles / Cartesian pose from the server. |
| **Hey Kara, status** | Prints currently running routine. |
| **Hey Kara, stop** | Requests a backend **software** stop and terminates the launched script. Plain **stop** is also accepted. |
| **Hey Kara, cancel** | Cancels an action awaiting confirmation. |
| **Hey Kara, help** | Prints available command examples. |

You can also pause after saying **Hey Kara** and give a command within six
seconds. For motion commands, speak the confirmation phrase within eight
seconds. The recognizer sometimes spells Kara as **Cara**; both are accepted.

**Example — seek yellow with real motion:**

1. Start `parol6-server` independently in another terminal, verify physical
   robot settings, correct camera calibration and calibrated drop locations.
2. Start the listener with `--enable-motion`.
3. Say: **“Hey Kara, seek yellow.”**
4. Listener prints the confirmation prompt. Say **“Confirm seek yellow.”**
5. The existing pick/drop script runs, opening its own OpenCV window.

## 4. Test without robot or microphone

```powershell
python -m unittest discover -s voice/tests -v
python -m voice --check-phrase "Hey Kara, seek yellow"
# Test multi-turn conversation with typed text (this CAN launch scripts):
python -m voice --text-mode
```

The `--check-phrase` option does not execute any subprocess. `--text-mode`
uses the regular actions, so use safe mode for initial testing.

## 5. Safety and limitations

- Robot motion is **disabled by default**. Never use `--enable-motion` without
  checking the physical workspace and keeping the hardware E-STOP accessible.
- Speech recognition is **not a safety-rated control system**. A software STOP
  may not stop a robot in all failure modes. Use the physical emergency stop.
- Only existing allowlisted command keys can run; recognized text is never
  passed to a shell. The microphone listener can control one launched script
  at a time.
- `seek_yellow.py` opens the USB camera independently, so stop the Qt GUI's
  camera preview before invoking it with voice. Do not run the GUI and voice
  controller against the same robot at the same time.
- The listener **does not start the PAROL6 server**. Start `parol6-server` first
  for real motion and `read pose`; dry-run also imports robot-related libraries.
- This is offline **speech-to-text + a fixed command parser**, not a newly
  trained wake-word neural network. Wake detection is approximate and can have
  false triggers or misses. No voice synthesis is included yet.
- The robot programs and vision packages must already work using the selected
  Python interpreter. The voice add-on does not install them.
