# Voice Latency Lab

A minimal **Flask + Twilio voice agent pipeline** for comparative LLM latency
research. The same voice pipeline runs against three swappable LLM backends so
you can measure time-to-first-token (TTFT), time-to-first-audio (TTFA), and
total turn latency across providers on identical utterances.

```
caller audio → Groq Whisper (STT) → LLM backend (streaming) → sentence-by-sentence TTS → audio back to caller
```

No RAG, no persona, no scheduling — just the raw pipeline for latency
measurement.

## LLM backends (swappable via one env var)

| `LLM_BACKEND` | Provider | Model (default) |
|---|---|---|
| `groq` | Groq chat completions (streaming) | `openai/gpt-oss-20b` |
| `gemini` | Google Gemini API (streaming) | `gemini-3.6-flash` |
| `gemma_local` | Ollama local API | `gemma2:2b-instruct-q4_0` |

All three implement the same interface in `llm_backend.py`:

```python
generate(prompt: str) -> Iterator[str]   # yields text chunks as generated
```

Switching backends is a one-line change in `.env` — nothing else in the
pipeline changes.

## Setup

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env     # then fill in your keys
```

Required keys/config in `.env`:

- `GROQ_API_KEY` — STT (Whisper) + the `groq` backend
- `GEMINI_API_KEY` — the `gemini` backend
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` — the `gemma_local` backend
  (install [Ollama](https://ollama.com), then `ollama pull gemma2:2b-instruct-q4_0`)
- `TTS_BACKEND` — `piper` (local neural TTS, needs `PIPER_MODEL`) or `null`
  (silence, for LLM-only latency)

## 1. Generate a comparison dataset (no phone needed)

```powershell
python test_backends.py                     # 5 default prompts x 3 backends
python test_backends.py --n 10 --tts        # 10 prompts, also measures TTS
python test_backends.py --backends groq gemini
```

Appends one row per prompt to `metrics.csv`, then:

```powershell
python app.py
# open http://localhost:5000/metrics
```

`/metrics` returns mean / median / min / max of `ttft_ms`, `ttfa_ms`,
`total_ms` grouped by backend (and by backend + prompt).

## 2. Live voice pipeline (Twilio)

1. Start the server: `python app.py`
2. Expose it: `ngrok http 5000` → note the URL, e.g. `https://abc.ngrok-free.dev`
3. Set `TWILIO_WEBHOOK_URL=https://abc.ngrok-free.dev/voice` in `.env`
4. **Inbound**: in the Twilio Console, set your number's *Voice → When a call
   comes in* webhook to the same URL.
5. **Outbound test call** (needs `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
   `TWILIO_FROM_NUMBER` in `.env`):

```powershell
python make_test_call.py --to +15551234567
```

Answer, speak, and the pipeline replies. Each turn is logged to `metrics.csv`.

## Metrics

Per turn, the CSV records:

```
call_id, backend, prompt,
t_audio_received, t_stt_done, t_llm_first_token, t_llm_done,
t_tts_first_byte, t_playback_start,
ttft_ms, ttfa_ms, total_ms
```

- **TTFT** = `t_llm_first_token − t_stt_done` (LLM time-to-first-token)
- **TTFA** = `t_tts_first_byte − t_audio_received` (time-to-first-audio)
- **Total** = `t_playback_start − t_audio_received` (full turn)

In `test_backends.py`, STT is not exercised, so `t_audio_received` /
`t_stt_done` are set to the request start — TTFT/TTFA are measured from prompt
submission.

## Tuning (env vars)

| Var | Default | Meaning |
|---|---|---|
| `SILENCE_THRESHOLD` | `300` | RMS above this counts as speech |
| `SILENCE_MS` | `700` | pause length that ends a turn |
| `MAX_TURN_MS` | `15000` | hard cap on one turn's audio |
| `GROQ_STT_MODEL` | `whisper-large-v3` | STT model |
| `PORT` | `5000` | Flask port |

## Files

```
app.py              Flask routes: /voice, /media-stream (WS), /metrics, /health
llm_backend.py      LLM interface + groq / gemini / gemma_local implementations
tts.py              TTS interface + piper / null implementations
audio.py            mu-law codec, resampling, WAV packaging, silence detection
metrics.py          thread-safe CSV logger + aggregation
test_backends.py    N prompts x backends -> metrics.csv
make_test_call.py   outbound Twilio call to trigger the pipeline
```