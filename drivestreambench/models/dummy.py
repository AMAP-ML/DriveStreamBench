"""Deterministic dummy model — used for smoke tests.

Has zero ML dependencies; runs on CPU instantly. Two shipped variants:

* :class:`AlwaysSilent` — emit ``"NO REPLY"`` every turn. Lower bound.
* :class:`AlertOnTurn` — emit ``"NO REPLY"`` until ``alert_turn``,
  then emit ``"ALERT"`` once, then ``"NO REPLY"`` thereafter.

These are sufficient to validate the protocol runner + scorer end-to-end
without touching any model weights or GPU code. Used by
``scripts/run_smoke_test.sh``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from drivestreambench.models.base import BaseModelAdapter


class AlwaysSilent(BaseModelAdapter):
    """Emit ``"NO REPLY"`` every turn — should score AlertScore=0 on positives,
    NSR=1 on negatives → Core Macro AlertScore = 0."""
    name = "dummy_always_silent"
    protocol = "native_streaming"

    def step(self, *, frames: List[Any], instruction: str,
             history: List[Dict[str, Any]], video_time: float) -> str:
        return "NO REPLY"


class AlertOnTurn(BaseModelAdapter):
    """Emit ``"ALERT"`` exactly at the configured turn, ``"NO REPLY"`` else."""

    def __init__(self, alert_turn: int = 5, name: str = "dummy_alert_on_turn"):
        self.alert_turn = alert_turn
        self.name = name
        self.protocol = "native_streaming"
        self._seen: int = 0

    def reset_session(self, session_id: str) -> None:
        self._seen = 0

    def step(self, *, frames: List[Any], instruction: str,
             history: List[Dict[str, Any]], video_time: float) -> str:
        out = "ALERT" if self._seen == self.alert_turn else "NO REPLY"
        self._seen += 1
        return out


class EchoInstruction(BaseModelAdapter):
    """Diagnostic — emit the watch instruction verbatim on turn 0, NO REPLY else.

    Useful for verifying that the runner threads ``instruction`` to the
    adapter on the first turn correctly.
    """
    name = "dummy_echo_instruction"
    protocol = "mode_alpha"

    def step(self, *, frames: List[Any], instruction: str,
             history: List[Dict[str, Any]], video_time: float) -> str:
        if not history:
            return instruction or "NO REPLY"
        return "NO REPLY"


__all__ = ["AlwaysSilent", "AlertOnTurn", "EchoInstruction"]
