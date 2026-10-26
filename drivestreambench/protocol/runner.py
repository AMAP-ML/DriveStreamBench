"""Protocol-agnostic session runner.

Drives a single watch-and-notify session by alternating:

  1. ``frame_sampler.sample_turn_frames`` — pick frames up to current decision time
  2. ``model.step(frames, instruction, history, video_time)`` — model emits a reply
  3. record reply + latency in :class:`StreamingSession`

Two protocol modes:

* ``native_streaming`` — model maintains KV cache across turns; ``step`` is
  cheap. Used by Qwen3-VL ladder, DriveStream-SFT.
* ``mode_alpha`` — stateless re-encoding per turn (closed-source APIs).

Both modes share the same outer loop. Mode is captured in the model
adapter's ``protocol`` attribute, not in the runner.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from drivestreambench.protocol.frame_sampler import sample_turn_frames
from drivestreambench.protocol.prompts import build_l4_user_message
from drivestreambench.protocol.session import StreamingSession, Turn, _Stopwatch


class SessionRunner:
    """Drive a single L4 streaming session given a model adapter."""

    def __init__(
        self,
        *,
        model: "Any",  # BaseModelAdapter (avoid hard import to keep modules independent)
        fps: int = 2,
        frames_per_turn: int = 2,
        decision_cadence_hz: float = 1.0,
        source_fps: float = 10.0,
    ) -> None:
        self.model = model
        self.fps = fps
        self.frames_per_turn = frames_per_turn
        self.decision_cadence_hz = decision_cadence_hz
        self.source_fps = source_fps

    def run(
        self,
        *,
        session_id: str,
        video_id: str,
        instruction: str,
        n_decisions: int,
        all_frame_paths: Optional[Sequence[str]] = None,
        n_total_frames: Optional[int] = None,
    ) -> StreamingSession:
        """Run all available frame groups up to ``n_decisions``.

        The scorer selects the first non-silent response; inference retains
        every decision step, including those following an alert.
        """
        if n_total_frames is None and all_frame_paths is not None:
            n_total_frames = len(all_frame_paths)
        sess = StreamingSession(
            session_id=session_id,
            instruction=instruction,
            video_id=video_id,
            fps=self.fps,
            frames_per_turn=self.frames_per_turn,
            decision_cadence_hz=self.decision_cadence_hz,
        )
        # Reset per-session state in the model adapter.
        if hasattr(self.model, "reset_session"):
            self.model.reset_session(session_id)
        history: List[dict] = []
        for i in range(n_decisions):
            window = sample_turn_frames(
                turn_idx=i,
                fps=self.fps,
                frames_per_turn=self.frames_per_turn,
                decision_cadence_hz=self.decision_cadence_hz,
                source_fps=self.source_fps,
                n_total_frames=n_total_frames,
            )
            if not window.frame_indices:
                break
            user_msg = build_l4_user_message(instruction) if i == 0 else ""
            frames_for_step: List[Any]
            if all_frame_paths is not None:
                frames_for_step = [
                    all_frame_paths[idx]
                    for idx in window.frame_indices
                    if 0 <= idx < len(all_frame_paths)
                ]
            else:
                frames_for_step = list(window.frame_indices)
            with _Stopwatch() as sw:
                reply = self.model.step(
                    frames=frames_for_step,
                    instruction=instruction,
                    history=history,
                    video_time=window.video_time,
                )
            sess.append_turn(
                Turn(
                    turn_idx=i,
                    video_time=window.video_time,
                    user_message=user_msg,
                    frame_indices=window.frame_indices,
                    reply=reply,
                    latency_ms=sw.elapsed_ms,
                )
            )
            history.append(
                {"turn_idx": i, "video_time": window.video_time, "reply": reply}
            )
        return sess


__all__ = ["SessionRunner"]
