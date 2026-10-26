"""Tau-curve helpers — compute AlertScore@τ for τ ∈ {1, 2, 3, 5} (paper §3.4.5).

Convenience module that re-exports the τ-grid-aware version of
:func:`drivestreambench.metrics.alertscore.score_l4`.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from drivestreambench.metrics.alertscore import DEFAULT_TAUS, score_l4


def score_l4_tau_grid(
    annotations: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    taus: Sequence[float] = DEFAULT_TAUS,
    *,
    fps: int = 2,
) -> Dict[str, Any]:
    """Same as :func:`score_l4` but emphasises that all τ are returned.

    Paper §3.4.5 asks for τ ∈ {1, 2, 3, 5}. The default :func:`score_l4`
    already evaluates all four; this is just a thin alias for callers
    that want to be explicit.
    """
    return score_l4(annotations, predictions, taus=taus, fps=fps)


__all__ = ["score_l4_tau_grid", "DEFAULT_TAUS"]
