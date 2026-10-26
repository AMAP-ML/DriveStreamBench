"""Streaming session object — holds turn history during a single
watch-and-notify episode.

Sessions are owned by :class:`drivestreambench.protocol.runner.SessionRunner`,
which orchestrates frame sampling → model step → reply collection over
``n_decisions`` turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """A single decision turn within a streaming session."""
    turn_idx: int
    video_time: float
    user_message: str
    frame_indices: List[int]
    reply: str
    latency_ms: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingSession:
    """In-flight L4 streaming session.

    Stays in-memory during a single video. Persisted to JSONL via
    :func:`to_prediction_record` at the end of the session.
    """
    session_id: str
    instruction: str
    video_id: str
    fps: int = 2
    frames_per_turn: int = 2
    decision_cadence_hz: float = 1.0
    turns: List[Turn] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def append_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def to_prediction_record(self, model_name: str = "") -> Dict[str, Any]:
        """Convert to the L4 prediction-JSONL schema."""
        return {
            "question_id": self.session_id,
            "model_name": model_name,
            "video_id": self.video_id,
            "fps": self.fps,
            "frames_per_turn": self.frames_per_turn,
            "decision_cadence_hz": self.decision_cadence_hz,
            "turns": [
                {
                    "turn_idx": t.turn_idx,
                    "video_time": t.video_time,
                    "reply": t.reply,
                    "latency_ms": round(t.latency_ms, 3),
                }
                for t in self.turns
            ],
            "extra": dict(self.extra),
        }


class _Stopwatch:
    """Tiny context manager → records elapsed milliseconds."""
    __slots__ = ("elapsed_ms", "_t0")

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "_Stopwatch":
        self._t0 = perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = (perf_counter() - self._t0) * 1000.0


__all__ = ["Turn", "StreamingSession", "_Stopwatch"]
