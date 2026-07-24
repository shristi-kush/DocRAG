"""Text-to-speech using Piper (local neural TTS, ONNX voices)."""

from __future__ import annotations

import logging
import subprocess
import sys
import wave
from pathlib import Path

from src.config import TTS_VOICE, TTS_VOICE_DIR

logger = logging.getLogger(__name__)

_voice = None


def _voice_onnx_path() -> Path:
    return TTS_VOICE_DIR / f"{TTS_VOICE}.onnx"


def ensure_voice() -> Path:
    """Ensure the Piper voice model is present, downloading it if needed."""
    onnx = _voice_onnx_path()
    if onnx.is_file():
        return onnx

    TTS_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Piper voice '%s' into %s ...", TTS_VOICE, TTS_VOICE_DIR)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "piper.download_voices",
                TTS_VOICE,
                "--data-dir",
                str(TTS_VOICE_DIR),
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Could not download Piper voice '{TTS_VOICE}'. Download it manually: "
            f"`python -m piper.download_voices {TTS_VOICE} --data-dir "
            f"{TTS_VOICE_DIR}`."
        ) from exc

    if not onnx.is_file():
        raise RuntimeError(
            f"Piper voice '{TTS_VOICE}' not found at {onnx} after download."
        )
    return onnx


def _get_voice():
    global _voice
    if _voice is None:
        from piper import PiperVoice

        onnx = ensure_voice()
        _voice = PiperVoice.load(str(onnx))
    return _voice


def synthesize(text: str, out_path: str | Path) -> Path:
    """Synthesize ``text`` to a WAV file and return its path."""
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    voice = _get_voice()
    with wave.open(str(out), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return out
