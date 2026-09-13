"""TTS backends.  All return 8 kHz mu-law bytes ready for Twilio Media Streams.

The interface is intentionally identical to the LLM backends:

    synthesize(text) -> bytes

Pipeline only calls this — never touches backend-specific code.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from audio import pcm16_to_mulaw, resample_pcm, silence_mulaw


class TTSBackend(ABC):
    name = "base"

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text into 8 kHz mu-law audio bytes."""
        raise NotImplementedError


class PiperTTS(TTSBackend):
    """Local neural TTS via the piper CLI (no API key needed).

    Install:  pip install piper-tts
    Voice:    download a .onnx model from https://github.com/rhasspy/piper
              (e.g. en_US-lessac-medium.onnx) and set PIPER_MODEL.
    """

    name = "piper"

    def __init__(
        self,
        model_path: Optional[str] = None,
        out_rate: int = 8000,
    ):
        self.model_path = model_path or os.getenv("PIPER_MODEL")
        if not self.model_path:
            raise RuntimeError(
                "PIPER_MODEL env var must point to a piper .onnx voice file. "
                "Download one from https://github.com/rhasspy/piper/releases"
            )
        self.out_rate = out_rate

    def synthesize(self, text: str) -> bytes:
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            subprocess.run(
                ["piper", "--model", self.model_path, "--output_file", tmp_path],
                input=text.encode("utf-8"),
                capture_output=True,
                check=True,
                timeout=60,
            )

            with wave.open(tmp_path, "rb") as w:
                rate = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

            if rate != self.out_rate:
                pcm = resample_pcm(pcm, rate, self.out_rate)

            return pcm16_to_mulaw(pcm)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


class NullTTS(TTSBackend):
    """Returns a short silence clip — use for measuring LLM-only latency."""

    name = "null"

    def synthesize(self, text: str) -> bytes:
        return silence_mulaw(0.1)


_TTS_BACKENDS = {
    "piper": PiperTTS,
    "null": NullTTS,
}


def get_tts(name: Optional[str] = None) -> TTSBackend:
    """Factory: TTS_BACKEND env var selects the implementation."""
    name = (name or os.getenv("TTS_BACKEND", "piper")).lower()
    if name not in _TTS_BACKENDS:
        raise ValueError(
            f"Unknown TTS_BACKEND {name!r}. Choose from: {sorted(_TTS_BACKENDS)}"
        )
    return _TTS_BACKENDS[name]()