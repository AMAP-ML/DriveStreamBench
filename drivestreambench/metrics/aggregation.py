"""Aggregation helpers — macro / micro across subcategories, fps, models."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def macro_mean(values: Sequence[float]) -> float:
    """Plain arithmetic mean. Returns 0.0 on empty input."""
    return sum(values) / len(values) if values else 0.0


def macro_over(per_cat: Dict[str, Dict[str, Any]], cats: Sequence[str], path: Sequence[str]) -> float:
    """Walk ``per_cat[cat][...path...]`` for each ``cat`` and macro-mean.

    Example::

        macro_over(per_cat, L4_CORE_SUBCATEGORIES, ["by_tau", "2.0", "alert_score"])
    """
    vals: List[float] = []
    for c in cats:
        node: Any = per_cat.get(c)
        if node is None:
            continue
        try:
            for k in path:
                node = node[k]
            vals.append(float(node))
        except (KeyError, TypeError, ValueError):
            continue
    return macro_mean(vals)


__all__ = ["macro_mean", "macro_over"]
