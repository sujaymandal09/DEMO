"""Flask + Twilio Media Streams voice pipeline for LLM latency research.

Flow per turn:
    caller audio -> Groq Whisper (STT) -> LLM backend (streaming)
    -> sentence-by-sentence TTS -> audio streamed back to the caller

Endpoints:
    POST /voice          Twilio webhook: answers the call, opens a Media Stream
    WS   /media-stream   Twilio Media Stream (real-time bidirectional audio)
    GET  /metrics        mean/median/min/max latency grouped by backend
    GET  /health         liveness + active backend

Run:  python app.py   (then expose via ngrok and point Twilio at /voice)
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_sock import Sock

from audio import mulaw_to_pcm16, pcm16_to_wav_bytes, rms
from llm_backend import get_backend
from metrics import MetricsLogger, compute_metrics, now_ms
from tts import get_tts

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

# --- tunables (env-overridable) ---
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "300"))
SILENCE_MS = float(os.getenv("SILENCE_MS", "700"))
MAX_TURN_MS = float(os.getenv("MAX_TURN_MS", "15000"))
STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_groq_client = None


# ---------------------------------------------------------------------------
# Stage 1 — STT (Groq Whisper)
# ---------------------------------------------------------------------------

def stt_transcribe(wav_bytes: bytes) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    result = _groq_client.audio.transcriptions.create(
        model=STT_MODEL,
        file=("turn.wav", wav_bytes, "audio/wav"),
    )
    return (result.text or "").strip()


# ---------------------------------------------------------------------------
# Sentence splitting + audio streaming back to Twilio
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> Tuple[List[str], str]:
    """Split on sentence boundaries; keep the trailing (incomplete) part."""
    parts = SENTENCE_RE.split(text)
    if len(parts) <= 1:
        return [], parts[0] if parts else ""
    return parts[:-1], parts[-1]


def send_media(ws, stream_sid: str, mulaw_bytes: bytes) -> None:
    ws.send(
        json.dumps(
            {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(mulaw_bytes).decode("ascii")},
            }
        )
    )


# ---------------------------------------------------------------------------
# One full turn: STT -> LLM (stream) -> TTS (sentence-by-sentence) -> send
# ---------------------------------------------------------------------------

def process_turn(
    ws,
    stream_sid: str,
    call_id: str,
    backend_name: str,
    mulaw_bytes: bytes,
    t_audio_received: float,
) -> None:
    logger = MetricsLogger()
    backend = get_backend(backend_name)
    tts = get_tts()

    # Stage 1 — STT
    pcm = mulaw_to_pcm16(mulaw_bytes)
    wav = pcm16_to_wav_bytes(pcm, 8000)
    transcript = stt_transcribe(wav)
    t_stt_done = now_ms()
    if not transcript:
        print(f"[{call_id}] empty transcript, skipping turn")
        return

    # Stage 2 + 3 — stream LLM tokens, TTS each completed sentence on the fly
    t_llm_first_token: Optional[float] = None
    t_llm_done: Optional[float] = None
    t_tts_first_byte: Optional[float] = None
    t_playback_start: Optional[float] = None
    remainder = ""

    for chunk in backend.generate(transcript):
        if t_llm_first_token is None:
            t_llm_first_token = now_ms()
        remainder += chunk
        sentences, remainder = split_sentences(remainder)
        for sentence in sentences:
            audio = tts.synthesize(sentence)
            if t_tts_first_byte is None:
                t_tts_first_byte = now_ms()
            send_media(ws, stream_sid, audio)
            if t_playback_start is None:
                t_playback_start = now_ms()

    t_llm_done = now_ms()

    # flush any trailing partial sentence
    if remainder.strip():
        audio = tts.synthesize(remainder)
        if t_tts_first_byte is None:
            t_tts_first_byte = now_ms()
        send_media(ws, stream_sid, audio)
        if t_playback_start is None:
            t_playback_start = now_ms()

    # --- per-call latency metrics ---
    ttft = (t_llm_first_token - t_stt_done) if t_llm_first_token else None
    ttfa = (t_tts_first_byte - t_audio_received) if t_tts_first_byte else None
    total = (t_playback_start or t_llm_done) - t_audio_received

    logger.log(
        {
            "call_id": call_id,
            "backend": backend.name,
            "prompt": transcript,
            "t_audio_received": t_audio_received,
            "t_stt_done": t_stt_done,
            "t_llm_first_token": t_llm_first_token,
            "t_llm_done": t_llm_done,
            "t_tts_first_byte": t_tts_first_byte,
            "t_playback_start": t_playback_start,
            "ttft_ms": ttft,
            "ttfa_ms": ttfa,
            "total_ms": total,
        }
    )
    print(
        f"[{call_id}] {backend.name}: "
        f"stt={t_stt_done - t_audio_received:.0f}ms "
        f"ttft={ttft:.0f}ms ttfa={ttfa:.0f}ms total={total:.0f}ms"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/voice", methods=["POST"])
def voice() -> Response:
    """Twilio webhook: answer the call and open a Media Stream."""
    base = os.getenv("PUBLIC_BASE_URL", f"wss://{request.host}")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    backend = os.getenv("LLM_BACKEND", "groq")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{base}/media-stream">
      <Parameter name="backend" value="{backend}"/>
    </Stream>
  </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@sock.route("/media-stream")
def media_stream(ws) -> None:
    """Twilio Media Stream: receive caller audio, stream TTS audio back."""
    stream_sid: Optional[str] = None
    call_id = "call-unknown"
    backend_name = os.getenv("LLM_BACKEND", "groq")

    buffer = bytearray()
    last_voice = time.time()
    turn_start: Optional[float] = None

    while True:
        message = ws.receive()
        if message is None:
            break
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            continue
        event = data.get("event")

        if event == "start":
            stream_sid = data.get("streamSid")
            call_id = data.get("start", {}).get("callSid", call_id)
            params = data.get("start", {}).get("customParameters", {}) or {}
            backend_name = params.get("backend") or backend_name
            continue

        if event == "media":
            payload = base64.b64decode(data["media"]["payload"])
            chunk_pcm = mulaw_to_pcm16(payload)
            if rms(chunk_pcm) > SILENCE_THRESHOLD:
                last_voice = time.time()
                if turn_start is None:
                    turn_start = now_ms()
            buffer += payload

            silence_ms = (time.time() - last_voice) * 1000.0
            turn_ms = (now_ms() - turn_start) if turn_start else 0.0
            if buffer and (silence_ms >= SILENCE_MS or turn_ms >= MAX_TURN_MS):
                try:
                    process_turn(
                        ws, stream_sid, call_id, backend_name,
                        bytes(buffer), turn_start,
                    )
                except Exception as exc:  # keep the socket alive on errors
                    print(f"[{call_id}] turn failed: {exc}")
                buffer.clear()
                turn_start = None

        elif event == "stop":
            if buffer and turn_start:
                try:
                    process_turn(
                        ws, stream_sid, call_id, backend_name,
                        bytes(buffer), turn_start,
                    )
                except Exception as exc:
                    print(f"[{call_id}] turn failed: {exc}")
            break


@app.route("/metrics")
def metrics():
    """Aggregate latency stats from the CSV, grouped by backend."""
    return jsonify(compute_metrics())


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "llm_backend": os.getenv("LLM_BACKEND", "groq"),
            "tts_backend": os.getenv("TTS_BACKEND", "piper"),
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )