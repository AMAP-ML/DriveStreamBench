"""Uniform, left-aligned frame sampling and causal decision times."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class FrameWindow:
    """Source-frame indices shown at one decision step."""

    turn_idx: int
    video_time: float
    frame_indices: List[int]
    source_fps: float


def source_frame_step(*, fps: int = 2, source_fps: float = 10.0) -> int:
    """Return the stride for the released source-frame grid."""
    if fps < 1 or source_fps <= 0 or fps > source_fps:
        raise ValueError("Require 1 <= fps <= source_fps")
    return max(1, int(round(source_fps / fps)))


def decision_count(
    n_total_frames: int,
    *,
    fps: int = 2,
    frames_per_turn: int = 2,
    source_fps: float = 10.0,
) -> int:
    """Count groups on the uniform grid, including a final partial group."""
    if n_total_frames < 0 or frames_per_turn < 1:
        raise ValueError("Require n_total_frames >= 0 and frames_per_turn >= 1")
    step = source_frame_step(fps=fps, source_fps=source_fps)
    n_selected = len(range(0, n_total_frames, step))
    return (n_selected + frames_per_turn - 1) // frames_per_turn


def sample_turn_frames(
    turn_idx: int,
    *,
    fps: int = 2,
    frames_per_turn: int = 2,
    decision_cadence_hz: float = 1.0,
    source_fps: float = 10.0,
    n_total_frames: int | None = None,
) -> FrameWindow:
    """Group the uniform source grid ``[0, step, 2*step, ...]``.

    At 2 FPS with two frames per turn, turn zero observes source frames
    0 and 5 and ends at decision time 1.0 s. Decision time is the number
    of processed sampled frames divided by FPS; a final singleton group
    therefore advances by 0.5 s. No annotation enters frame selection.
    """
    if turn_idx < 0 or frames_per_turn < 1:
        raise ValueError("Require turn_idx >= 0 and frames_per_turn >= 1")
    step = source_frame_step(fps=fps, source_fps=source_fps)
    if decision_cadence_hz <= 0 or not math.isclose(
        frames_per_turn / fps, 1.0 / decision_cadence_hz
    ):
        raise ValueError("frames_per_turn / fps must match the decision interval")
    start = turn_idx * frames_per_turn
    indices = [i * step for i in range(start, start + frames_per_turn)]
    if n_total_frames is not None:
        if n_total_frames < 0:
            raise ValueError("n_total_frames must be >= 0")
        indices = [i for i in indices if i < n_total_frames]
    return FrameWindow(
        turn_idx=turn_idx,
        video_time=(start + len(indices)) / fps,
        frame_indices=indices,
        source_fps=source_fps,
    )


def sample_session_grid(
    n_decisions: int,
    *,
    fps: int = 2,
    frames_per_turn: int = 2,
    decision_cadence_hz: float = 1.0,
    source_fps: float = 10.0,
    n_total_frames: int | None = None,
) -> List[FrameWindow]:
    """Return nonempty frame groups up to ``n_decisions``."""
    grid = []
    for i in range(n_decisions):
        window = sample_turn_frames(
            turn_idx=i,
            fps=fps,
            frames_per_turn=frames_per_turn,
            decision_cadence_hz=decision_cadence_hz,
            source_fps=source_fps,
            n_total_frames=n_total_frames,
        )
        if not window.frame_indices:
            break
        grid.append(window)
    return grid


__all__ = [
    "FrameWindow",
    "decision_count",
    "sample_turn_frames",
    "sample_session_grid",
    "source_frame_step",
]
