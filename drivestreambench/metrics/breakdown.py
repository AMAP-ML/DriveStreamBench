"""Per-subcategory / per-alert-mode / per-tau breakdown helpers.

These produce tabular views from the output of
:func:`drivestreambench.metrics.alertscore.score_l4`.
"""
from __future__ import annotations

from typing import Any, Dict, List

from drivestreambench.data.schemas import L4_CORE_SUBCATEGORIES, L4_OOD_SUBCATEGORY


def per_cat_table(score_result: Dict[str, Any], tau: float = 2.0) -> List[Dict[str, Any]]:
    """One row per subcategory: cat / n_pos / n_neg / hit / nsr / alert_score@τ."""
    pcat = score_result.get("per_subcategory", {})
    rows: List[Dict[str, Any]] = []
    for cat in (*L4_CORE_SUBCATEGORIES, L4_OOD_SUBCATEGORY):
        if cat not in pcat:
            continue
        b = pcat[cat]
        t = b["by_tau"].get(str(tau), {})
        rows.append({
            "subcategory": cat,
            "n_pos": b["n_pos"],
            "n_neg": b["n_neg"],
            "hit": t.get("hit_rate", 0.0),
            "nsr": b.get("nsr", 0.0),
            f"alert_score@{tau}s": t.get("alert_score", 0.0),
        })
    return rows


def tau_curve(score_result: Dict[str, Any], cat: str) -> List[Dict[str, Any]]:
    """Per-τ rows for a single subcategory."""
    pcat = score_result.get("per_subcategory", {})
    block = pcat.get(cat, {})
    rows: List[Dict[str, Any]] = []
    for tau_str, t in block.get("by_tau", {}).items():
        rows.append({
            "tau": float(tau_str),
            "hit": t["hit_rate"],
            "alert_score": t["alert_score"],
        })
    return rows


__all__ = ["per_cat_table", "tau_curve"]
