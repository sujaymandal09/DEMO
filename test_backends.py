"""Send the same N prompts to every LLM backend and append results to the
metrics CSV — one command generates a full comparison dataset.

Usage:
    python test_backends.py                  # 5 default prompts x 3 backends
    python test_backends.py --n 10           # cycle the default list to 10
    python test_backends.py --prompts "Q1" "Q2" --backends groq gemini
    python test_backends.py --tts            # also measure TTS first-byte time

Notes:
    - STT is not exercised here; t_audio_received / t_stt_done are set to the
      request start so TTFT/TTFA are measured from prompt submission.
    - Requires the API keys in .env (or environment) for the backends tested.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import List, Optional

from dotenv import load_dotenv

from llm_backend import BACKENDS, get_backend
from metrics import MetricsLogger, now_ms
from tts import get_tts

DEFAULT_PROMPTS: List[str] = [
    "Explain the difference between TCP and UDP in two sentences.",
    "What is the capital of France and what is it famous for?",
    "Write a haiku about a rainy day.",
    "List three benefits of regular exercise.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts", nargs="*", default=None,
        help="sample prompts (default: built-in list of 5)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="total number of prompts (cycles the prompt list to reach N)",
    )
    parser.add_argument(
        "--backends", nargs="*", default=None,
        help=f"backends to test (default: all of {sorted(BACKENDS)})",
    )
    parser.add_argument(
        "--csv", default=os.getenv("METRICS_CSV", "metrics.csv"),
        help="metrics CSV path",
    )
    parser.add_argument(
        "--tts", action="store_true",
        help="also measure TTS time-to-first-audio (needs TTS_BACKEND configured)",
    )
    args = parser.parse_args()

    prompts: List[str] = list(args.prompts) if args.prompts else list(DEFAULT_PROMPTS)
    if args.n:
        prompts = (prompts * (args.n // len(prompts) + 1))[: args.n]
    backends: List[str] = args.backends or sorted(BACKENDS)

    logger = MetricsLogger(args.csv)
    tts = get_tts() if args.tts else None

    for backend_name in backends:
        try:
            backend = get_backend(backend_name)
        except Exception as exc:
            print(f"!! backend {backend_name} unavailable: {exc}", file=sys.stderr)
            continue
        print(f"\n=== backend: {backend_name} ===")
        for prompt in prompts:
            call_id = f"test-{backend_name}-{uuid.uuid4().hex[:8]}"
            t0 = now_ms()
            first_token: Optional[float] = None
            parts: List[str] = []
            try:
                for chunk in backend.generate(prompt):
                    if first_token is None:
                        first_token = now_ms()
                    parts.append(chunk)
            except Exception as exc:
                print(f"  [{prompt[:40]}...] FAILED: {exc}", file=sys.stderr)
                continue
            t_done = now_ms()
            text = "".join(parts)
            ttft = (first_token - t0) if first_token else None

            tts_first: Optional[float] = None
            if tts and text.strip():
                tts.synthesize(text)
                tts_first = now_ms()

            logger.log(
                {
                    "call_id": call_id,
                    "backend": backend_name,
                    "prompt": prompt,
                    "t_audio_received": t0,
                    "t_stt_done": t0,
                    "t_llm_first_token": first_token,
                    "t_llm_done": t_done,
                    "t_tts_first_byte": tts_first,
                    "t_playback_start": tts_first,
                    "ttft_ms": ttft,
                    "ttfa_ms": (tts_first - t0) if tts_first else "",
                    "total_ms": t_done - t0,
                }
            )
            print(
                f"  ttft={ttft:7.1f}ms  total={t_done - t0:7.1f}ms  "
                f"chars={len(text):4d}  | {prompt[:50]}"
            )

    print(
        f"\nDone. Results appended to {args.csv}. "
        f"Start the server (python app.py) and open /metrics to see aggregates."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())