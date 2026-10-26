"""Latency and real-time factor (RTF) summaries for L4 predictions.

Inputs are prediction JSONL records with ``turns[].latency_ms`` and an
explicit decision cadence in Hz. Summaries include pooled per-decision
latency statistics and mean RTF relative to the decision interval.

``k_star`` is the first 1-based position in the per-session latency
sequences whose across-session median RTF exceeds 1.0. Only sessions
with a latency entry at that position contribute; this is not a
cumulative statistic over earlier decisions.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from drivestreambench.data.io import iter_jsonl, write_json


def per_session_latencies(prediction_jsonl: str | Path) -> List[List[float]]:
    """Return list of per-session latency_ms lists (None entries dropped)."""
    out: List[List[float]] = []
    for rec in iter_jsonl(prediction_jsonl):
        latencies = []
        for t in rec.get("turns") or []:
            v = t.get("latency_ms")
            if v is None:
                continue
            try:
                latencies.append(float(v))
            except (TypeError, ValueError):
                continue
        if latencies:
            out.append(latencies)
    return out


def aggregate_latencies(
    per_session: Sequence[Sequence[float]],
    *,
    decision_cadence_hz: float = 1.0,
) -> Dict[str, Any]:
    """Compute median/p95/max/mean latency + RTF aggregates."""
    flat = [v for s in per_session for v in s]
    if not flat:
        return {
            "n_sessions": 0,
            "n_turns": 0,
            "median_ms": None, "p95_ms": None, "max_ms": None, "mean_ms": None,
            "rtf_mean": None,
            "decision_cadence_hz": decision_cadence_hz,
        }
    flat.sort()
    n = len(flat)
    p50 = flat[n // 2]
    p95 = flat[min(n - 1, int(0.95 * n))]
    return {
        "n_sessions": len(per_session),
        "n_turns": n,
        "median_ms": p50,
        "p95_ms": p95,
        "max_ms": flat[-1],
        "mean_ms": statistics.fmean(flat),
        "rtf_mean": (statistics.fmean(flat) / 1000.0) / (1.0 / decision_cadence_hz),
        "decision_cadence_hz": decision_cadence_hz,
    }


def find_k_star(
    per_session: Sequence[Sequence[float]],
    *,
    decision_cadence_hz: float = 1.0,
    threshold_rtf: float = 1.0,
    max_k: int = 200,
) -> Optional[int]:
    """First 1-based latency index whose median RTF exceeds ``threshold_rtf``.

    At index k, collect entry k - 1 from each session with at least k
    entries. The sorted upper-middle latency defines the median RTF.
    Earlier entries are not accumulated. Return None if no checked
    index exceeds the threshold.
    """
    interval_s = 1.0 / decision_cadence_hz
    for k in range(1, max_k + 1):
        per_session_at_k = [s[k - 1] for s in per_session if len(s) >= k]
        if not per_session_at_k:
            return None
        per_session_at_k.sort()
        median_ms = per_session_at_k[len(per_session_at_k) // 2]
        rtf = (median_ms / 1000.0) / interval_s
        if rtf > threshold_rtf:
            return k
    return None


def measure_from_prediction(
    prediction_jsonl: str | Path,
    *,
    decision_cadence_hz: float = 1.0,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """High-level: compute the full latency report for one prediction file."""
    sessions = per_session_latencies(prediction_jsonl)
    agg = aggregate_latencies(sessions, decision_cadence_hz=decision_cadence_hz)
    k_star = find_k_star(sessions, decision_cadence_hz=decision_cadence_hz)
    report = {
        "prediction_jsonl": str(prediction_jsonl),
        "decision_cadence_hz": decision_cadence_hz,
        "median_ms": agg["median_ms"],
        "p95_ms": agg["p95_ms"],
        "max_ms": agg["max_ms"],
        "mean_ms": agg["mean_ms"],
        "rtf_mean": agg["rtf_mean"],
        "k_star": k_star,
        "n_sessions": agg["n_sessions"],
        "n_turns": agg["n_turns"],
    }
    if output_path:
        write_json(output_path, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute latency / RTF / k* from prediction JSONL")
    ap.add_argument("--prediction", required=True, help="prediction JSONL with per-turn latency_ms")
    ap.add_argument("--cadence-hz", type=float, default=1.0)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    report = measure_from_prediction(
        args.prediction,
        decision_cadence_hz=args.cadence_hz,
        output_path=args.output or None,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
