"""Run from repository root: python -m voice."""
from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path

from voice.assistant import KaraAssistant
from voice.runner import ProgramRunner

VOICE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = VOICE_ROOT / "models" / "vosk-model-small-en-us-0.15"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KARA standalone voice commands (offline Vosk)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to an extracted Vosk model")
    p.add_argument("--device", type=int, default=None, help="Input microphone device index")
    p.add_argument("--camera", type=int, default=0, help="Camera index passed to vision scripts")
    p.add_argument("--python", default=None, help="Python executable for PAROL6 robot scripts")
    p.add_argument("--enable-motion", action="store_true",
                   help="Opt in to robot movement; still requires a spoken confirmation")
    p.add_argument("--list-devices", action="store_true", help="Show audio input devices and exit")
    p.add_argument("--text-mode", action="store_true", help="Type test phrases instead of microphone audio")
    p.add_argument("--check-phrase", metavar="WORDS",
                   help="Test one phrase without audio or executing robot commands")
    return p


def run_audio(assistant: KaraAssistant, model_path: Path, device: int | None) -> None:
    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Run: python -m pip install -r voice/requirements.txt") from exc

    if not (model_path / "conf").is_dir() or not (model_path / "am").is_dir():
        raise SystemExit(
            f"Speech model missing at {model_path}\n"
            "Download it by running: python -m voice.setup_model"
        )

    SetLogLevel(-1)
    model = Model(str(model_path))
    # Match the hardware's supported input sample rate. Vosk handles conversion
    # to the model's required rate; this avoids 16 kHz PortAudio errors on
    # devices that only expose 44.1/48 kHz (common on Windows).
    try:
        sample_rate = int(sd.query_devices(device, "input")["default_samplerate"])
    except (ValueError, sd.PortAudioError) as exc:
        raise SystemExit(f"No usable microphone input: {exc}. Run --list-devices.") from exc
    rec = KaldiRecognizer(model, sample_rate)
    samples: queue.Queue[bytes] = queue.Queue(maxsize=20)

    def on_audio(indata, frames, time_info, status):
        if status:
            # The callback must be short and must never block the microphone.
            print(f"[AUDIO] {status}", file=sys.stderr)
        try:
            samples.put_nowait(bytes(indata))
        except queue.Full:
            pass

    print(f"[KARA] Offline voice model loaded at {sample_rate} Hz. Say 'Hey Kara, help'.")
    print("[KARA] Say 'Hey Kara, stop' to request a software stop. Ctrl+C exits.")
    print("[KARA] Robot movement is " + ("ENABLED (confirmation required)." if assistant.enable_motion
                                           else "DISABLED (seek yellow runs dry-run)."))
    try:
        with sd.RawInputStream(samplerate=sample_rate, blocksize=4000,
                               device=device, dtype="int16", channels=1, callback=on_audio):
            while True:
                try:
                    data = samples.get(timeout=0.5)
                except queue.Empty:
                    assistant.runner.running()
                    continue
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get("text", "").strip()
                    if text:
                        print(f"[HEARD] {text}", flush=True)
                        answer = assistant.handle(text)
                        if answer:
                            print(f"[KARA] {answer}", flush=True)
                assistant.runner.running()  # Report completed scripts.
    except KeyboardInterrupt:
        print("\n[KARA] Listener exiting.")
    except sd.PortAudioError as exc:
        raise SystemExit(f"Cannot open microphone: {exc}. Run --list-devices.") from exc
    finally:
        assistant.runner.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SystemExit("Install audio dependencies with: python -m pip install -r voice/requirements.txt") from exc
        print(sd.query_devices())
        return 0
    if not 0 <= args.camera <= 20:
        raise SystemExit("--camera must be between 0 and 20")
    runner = ProgramRunner(python=args.python, camera=args.camera, preview=args.check_phrase is not None)
    assistant = KaraAssistant(runner, enable_motion=args.enable_motion)
    if args.check_phrase is not None:
        print("[HEARD] " + args.check_phrase)
        print("[KARA] " + (assistant.handle(args.check_phrase) or "Ignored: no wake phrase."))
        return 0
    if args.text_mode:
        print("[KARA] Text test mode. Type 'Hey Kara, help' or Ctrl+C to exit.")
        try:
            while True:
                try:
                    line = input("speech> ")
                except EOFError:
                    break
                answer = assistant.handle(line)
                print("[KARA] " + (answer or "Ignored: no wake phrase."))
        except KeyboardInterrupt:
            print("\n[KARA] Exiting.")
        finally:
            runner.close()
        return 0
    run_audio(assistant, args.model, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
