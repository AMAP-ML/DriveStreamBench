"""Shared ``--model`` spec resolver for inference entry points.

Both :mod:`drivestreambench.inference.run_l4` (streaming watch-and-notify)
and :mod:`drivestreambench.inference.run_l123` (reactive QA) accept a
``--model <spec>`` flag.  The spec syntax is the same across both
entries; this module is the single resolver.

Supported spec syntax
---------------------
``dummy`` / ``always_silent`` / ``silent``
    Always-NO-REPLY dummy (smoke test).  L1-L3 will receive
    ``"NO REPLY"`` strings; the reactive scorer treats them as wrong
    answers, which is the intended sanity check.
``dummy_yes`` / ``dummy_no``
    Reactive dummies that always answer "Yes" / "No".  Useful for
    L1/L3 majority-class baselines.
``alert_on_turn:N``
    Emit ALERT exactly at turn ``N`` (smoke test for L4 window logic).
``api_vlm`` / ``api_vlm:<model_name>``
    Mode-α stateless API adapter.  Reads ``LLM_API_KEY`` /
    ``LLM_BASE_URL`` / ``LLM_MODEL_NAME`` from environment.
``qwen3vl:<checkpoint_path>``
    Native-streaming Qwen3-VL via vLLM (lazy-imported so the package
    stays importable on CPU-only machines).
External streaming baselines run through their native batch entry point:
``python external_baselines/run.py --help``.
"""

from __future__ import annotations

from typing import List

from drivestreambench.models.base import BaseModelAdapter


def resolve_model(spec: str) -> BaseModelAdapter:
    spec = (spec or "").strip()
    if not spec:
        raise SystemExit("--model spec must be non-empty.")
    head = spec.split(":", 1)[0].lower()

    if head in ("dummy", "always_silent", "silent"):
        from drivestreambench.models.dummy import AlwaysSilent

        return AlwaysSilent()

    if head == "dummy_yes":
        return _ConstReactive(name="dummy_yes", reply="Yes")

    if head == "dummy_no":
        return _ConstReactive(name="dummy_no", reply="No")

    if head == "alert_on_turn":
        try:
            n = int(spec.split(":", 1)[1])
        except (IndexError, ValueError):
            n = 5
        from drivestreambench.models.dummy import AlertOnTurn

        return AlertOnTurn(alert_turn=n)

    if head == "api_vlm":
        from drivestreambench.models.api_vlm import APIVLM

        parts = spec.split(":", 1)
        model_name = parts[1] if len(parts) > 1 else None
        return APIVLM(model_name=model_name)

    if head == "qwen3vl":
        from drivestreambench.models.qwen3vl import Qwen3VLAdapter

        parts = spec.split(":", 1)
        if len(parts) < 2:
            raise SystemExit(
                "qwen3vl spec must include a checkpoint path: 'qwen3vl:<path>'"
            )
        return Qwen3VLAdapter(model_path=parts[1])

    from drivestreambench.models.streaming_baselines import ADAPTERS

    if head in ADAPTERS:
        raise SystemExit(
            f"{head} uses native batch inference. Run "
            "'python external_baselines/run.py --help' for its command."
        )

    known: List[str] = [
        "dummy",
        "dummy_yes",
        "dummy_no",
        "alert_on_turn:N",
        "api_vlm",
        "qwen3vl:<path>",
    ]
    raise SystemExit(
        f"Unknown --model spec: {spec!r}.  Try one of:\n  - " + "\n  - ".join(known)
    )


class _ConstReactive(BaseModelAdapter):
    """Reactive dummy that returns a constant string for every step.

    Used by ``--model dummy_yes`` / ``--model dummy_no`` for L1/L3
    majority-class smoke baselines.
    """

    protocol: str = "mode_alpha"

    def __init__(self, *, name: str, reply: str) -> None:
        self.name = name
        self._reply = reply

    def step(self, *, frames, instruction, history, video_time):
        return self._reply


__all__ = ["resolve_model"]
