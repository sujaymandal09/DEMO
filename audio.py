"""Audio helpers: mu-law codec, resampling, WAV packaging, silence detection.

Twilio Media Streams deliver/accept audio as base64-encoded 8 kHz mu-law
(G.711). Everything in this module converts between that wire format and
16-bit PCM numpy arrays.
"""
from __future__ import annotations

import io
import wave

import numpy as np

MULAW_BIAS = 132
MULAW_MAX = 32635


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    """Encode 16-bit PCM samples to 8-bit mu-law bytes (RFC 3551)."""
    pcm = np.asarray(pcm, dtype=np.int32)
    sign = (pcm >> 8) & 0x80
    mag = np.minimum(np.abs(pcm), MULAW_MAX) + MULAW_BIAS
    exp = np.zeros_like(mag)
    for e in range(7, 0, -1):
        exp += (mag >= (1 << (e + 7))).astype(np.int32)
    mant = (mag >> (exp + 3)) & 0x0F
    out = ~(sign | (exp << 4) | mant) & 0xFF
    return out.astype(np.uint8).tobytes()


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    """Decode mu-law bytes to 16-bit PCM samples."""
    u = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    u = ~u & 0xFF
    sign = (u & 0x80) != 0
    exp = (u >> 4) & 0x07
    mant = u & 0x0F
    sample = ((mant << 3) + MULAW_BIAS) << exp
    sample -= MULAW_BIAS
    sample = np.where(sign, -sample, sample)
    return sample.astype(np.int16)


def resample_pcm(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample (good enough for voice)."""
    if src_rate == dst_rate:
        return pcm
    n_out = max(1, int(round(len(pcm) * dst_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm.astype(np.float64)).astype(np.int16)


def pcm16_to_wav_bytes(pcm: np.ndarray, rate: int) -> bytes:
    """Package 16-bit PCM into a WAV file in memory (for STT uploads)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


def rms(pcm: np.ndarray) -> float:
    """Root-mean-square amplitude of a PCM buffer (silence detector input)."""
    if len(pcm) == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))


def silence_mulaw(seconds: float, rate: int = 8000) -> bytes:
    """Generate `seconds` of digital silence as mu-law bytes."""
    n = max(1, int(seconds * rate))
    return pcm16_to_mulaw(np.zeros(n, dtype=np.int16))