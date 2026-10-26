"""Batch interfaces for models with their own native streaming runtimes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from drivestreambench.models.base import BaseModelAdapter


def normalize_to_alert_or_no_reply(raw: str | None) -> str:
    """Retain native text for score-time parsing; None denotes silence.

    Free-form ``NO`` remains an action-level response. The downstream parser
    decides whether each reply is ALERT, NO REPLY or INVALID.
    """
    return "" if raw is None else str(raw)


class ExternalStreamingBaseline(BaseModelAdapter):
    """Run an input file through a model's native batch inference runtime."""

    yaml_name: str = ""
    protocol: str = "native_streaming"

    def __init__(self, *, install_dir: str, checkpoint: str, **kwargs: Any) -> None:
        if not self.yaml_name:
            raise ValueError("An external baseline name is required")
        self.name = self.yaml_name
        self.install_dir = Path(install_dir)
        self.checkpoint = checkpoint
        self.options = kwargs

    def run(self, *, input_path: str, gold_path: str, output_path: str, **kwargs: Any) -> list[str]:
        from external_baselines.native import run_native

        return run_native(self.yaml_name, install_dir=self.install_dir, checkpoint=self.checkpoint,
            input_path=Path(input_path), gold_path=Path(gold_path), output_path=Path(output_path),
            **{**self.options, **kwargs})

    def reset_session(self, session_id: str) -> None:
        raise NotImplementedError(f"{self.name} uses native batch inference. Use external_baselines/run.py {self.name} or adapter.run().")

    def step(self, *, frames: list[Any], instruction: str, history: list[dict[str, Any]], video_time: float) -> str:
        raise NotImplementedError(f"{self.name} uses native batch inference. Use external_baselines/run.py {self.name} or adapter.run().")


__all__ = ["ExternalStreamingBaseline", "normalize_to_alert_or_no_reply"]
