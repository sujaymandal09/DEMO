"""CSV metrics logging + aggregation for latency comparison."""
from __future__ import annotations

import csv
import os
import threading
import time
from typing import Any, Dict, List, Optional

METRICS_COLUMNS: List[str] = [
    "call_id",
    "backend",
    "prompt",
    "t_audio_received",
    "t_stt_done",
    "t_llm_first_token",
    "t_llm_done",
    "t_tts_first_byte",
    "t_playback_start",
    "ttft_ms",
    "ttfa_ms",
    "total_ms",
]


def now_ms() -> float:
    """Wall-clock time in milliseconds (epoch)."""
    return time.time() * 1000.0


class MetricsLogger:
    """Thread-safe CSV appender."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("METRICS_CSV", "metrics.csv")
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=METRICS_COLUMNS).writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
                writer.writerow({c: row.get(c, "") for c in METRICS_COLUMNS})


# ---------------------------------------------------------------------------
# Aggregation for the /metrics endpoint
# ---------------------------------------------------------------------------

def _f(v: Any) -> Optional[float]:
    """Coerce to float; return None on failure."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stats(values: List[float]) -> Optional[Dict[str, Any]]:
    """Compute count / mean / median / min / max for a list of floats."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    values.sort()
    n = len(values)
    median = (
        values[n // 2]
        if n % 2
        else (values[n // 2 - 1] + values[n // 2]) / 2
    )
    return {
        "count": n,
        "mean_ms": round(sum(values) / n, 2),
        "median_ms": round(median, 2),
        "min_ms": round(values[0], 2),
        "max_ms": round(values[-1], 2),
    }


def compute_metrics(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the CSV and return aggregates grouped by backend (and optionally
    by backend + prompt)."""
    path = path or os.getenv("METRICS_CSV", "metrics.csv")
    rows: List[Dict[str, str]] = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    by_backend: Dict[str, Dict[str, List[float]]] = {}
    by_pair: Dict[str, Dict[str, List[float]]] = {}

    for r in rows:
        backend = (r.get("backend") or "").strip()
        prompt = (r.get("prompt") or "").strip()
        if not backend:
            continue
        for key in ("ttft_ms", "ttfa_ms", "total_ms"):
            v = _f(r.get(key))
            by_backend.setdefault(backend, {}).setdefault(key, []).append(v)
            if prompt:
                pair_key = f"{backend} | {prompt[:80]}"
                by_pair.setdefault(pair_key, {}).setdefault(key, []).append(v)

    def _agg(d: Dict[str, List[float]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, vals in d.items():
            s = _stats(vals)
            if s:
                out[key] = s
        return out

    return {
        "by_backend": {b: _agg(m) for b, m in by_backend.items()},
        "by_backend_prompt": {p: _agg(m) for p, m in by_pair.items()},
    }