"""Explicit one-time download of the official Vosk English speech model."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VOICE_ROOT = Path(__file__).resolve().parent
NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
# Published model artifact checksum (matches the Rhasspy/VoskModels release).
SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
DEST = VOICE_ROOT / "models" / NAME


def download() -> Path:
    if (DEST / "am").is_dir() and (DEST / "conf").is_dir():
        print(f"Already installed: {DEST}")
        return DEST
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kara-vosk-") as temp:
        temp_dir = Path(temp)
        archive = temp_dir / (NAME + ".zip")
        digest = hashlib.sha256()
        print(f"Downloading ~40 MB from {URL} ...", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as res, archive.open("wb") as out:
            while chunk := res.read(256 * 1024):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("Downloaded model checksum mismatch. Refusing to extract it.")
        with zipfile.ZipFile(archive) as z:
            # Refuse archives with unexpected content, paths, or huge expansion.
            total = 0
            for member in z.infolist():
                item = Path(member.filename)
                if (item.is_absolute() or ".." in item.parts or not item.parts or
                        item.parts[0] != NAME):
                    raise RuntimeError("Unexpected path in model ZIP archive")
                total += member.file_size
                if total > 300 * 1024 * 1024:
                    raise RuntimeError("Model ZIP exceeds expected size")
            z.extractall(temp_dir)
        extracted = temp_dir / NAME
        if not (extracted / "am").is_dir() or not (extracted / "conf").is_dir():
            raise RuntimeError("Model archive missing required data")
        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.move(str(extracted), str(DEST))
    print(f"Installed offline speech model: {DEST}")
    return DEST


if __name__ == "__main__":
    download()
