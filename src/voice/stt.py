"""Speech-to-text using faster-whisper (fully local, CPU-friendly)."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import STT_COMPUTE_TYPE, STT_DEVICE, STT_MODEL_SIZE

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model '%s' (%s).", STT_MODEL_SIZE, STT_COMPUTE_TYPE)
        _model = WhisperModel(
            STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE
        )
    return _model


def transcribe(audio_path: str | Path) -> str:
    """Transcribe an audio file to text.

    Accepts any format faster-whisper/ffmpeg can decode (wav, mp3, m4a, ...).
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = _get_model()
    segments, _info = model.transcribe(str(path))
    text = " ".join(segment.text for segment in segments).strip()
    return text
