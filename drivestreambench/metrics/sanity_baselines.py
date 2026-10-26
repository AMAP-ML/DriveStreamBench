"""Non-visual shortcut baselines (paper §3.4.2 sanity check).

Five baselines that *do not* observe video frames:

* **Always Silent** — emit ``NO REPLY`` every turn. NSR=1, Hit=0 → AlertScore=0.
* **Alert at First Turn** — emit ``ALERT`` at turn 0, then ``NO REPLY``.
* **Fixed Absolute Time** — emit ``ALERT`` at turn ``round(α_global)``.
* **Category Temporal Prior** — per-subcategory ``α_c``.
* **Positive-oracle Temporal Prior** — same as Category Temporal Prior, but
  only fires on positive sessions (oracle gating). Diagnostic only — not
  a deployable baseline. Used to show the upper bound of timing-only
  knowledge.

The first four are deployable (no gold leakage) and *must* score
near-zero on AlertScore by design — that's the sanity check the paper
makes against shortcut solutions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from drivestreambench.data.schemas import L4_ALL_SUBCATEGORIES
from drivestreambench.metrics.alertscore import score_l4


def _make_pred_for(qid: str, alert_at_turn: int | None, n_turns: int = 60) -> Dict[str, Any]:
    """Helper — synthesise a prediction with one ALERT at ``alert_at_turn``."""
    turns = []
    for i in range(n_turns):
        reply = "ALERT" if i == alert_at_turn else "NO REPLY"
        turns.append({"turn_idx": i, "video_time": float(i + 1), "reply": reply})
    return {"question_id": qid, "model_name": "shortcut", "turns": turns}


def _alpha_per_cat(annotations: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute per-subcategory mean ``t_ref`` from training-split positives.

    Caller should pass training-set annotations; this implementation
    accepts whatever annotations are given (so users can sanity check
    on test-set if they want, but the paper convention is train-derived).
    """
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for ann in annotations:
        if not ann.get("is_positive"):
            continue
        t = ann.get("t_ref")
        if t is None:
            continue
        sums[ann["subcategory"]] += float(t)
        counts[ann["subcategory"]] += 1
    return {c: sums[c] / counts[c] for c in counts if counts[c] > 0}


def _alpha_global(annotations: Sequence[Dict[str, Any]]) -> float:
    """Cross-category mean ``t_ref`` of positive sessions."""
    pos = [float(a["t_ref"]) for a in annotations if a.get("is_positive") and a.get("t_ref") is not None]
    return sum(pos) / len(pos) if pos else 0.0


def shortcut_baselines(
    test_annotations: Sequence[Dict[str, Any]],
    train_annotations: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute all 5 shortcut-baseline AlertScores against the test set.

    Returns a dict ``baseline_name -> score_l4(...) result``. The macro
    Core AlertScore@2 is the headline number; full per-cat breakdown is
    nested inside each value.
    """
    train_anns = list(train_annotations) if train_annotations is not None else list(test_annotations)
    alpha_per = _alpha_per_cat(train_anns)
    alpha_glob = _alpha_global(train_anns)

    out: Dict[str, Dict[str, Any]] = {}

    # Always Silent — never alert
    silent_preds = [
        {"question_id": a["question_id"], "model_name": "always_silent",
         "turns": [{"turn_idx": i, "video_time": float(i + 1), "reply": "NO REPLY"} for i in range(60)]}
        for a in test_annotations
    ]
    out["always_silent"] = score_l4(test_annotations, silent_preds)

    # Alert at First Turn — alert at turn 0
    first_preds = [_make_pred_for(a["question_id"], 0) for a in test_annotations]
    out["alert_first_turn"] = score_l4(test_annotations, first_preds)

    # Fixed Absolute Time — α global
    glob_turn = int(round(alpha_glob)) if alpha_glob > 0 else 0
    fixed_preds = [_make_pred_for(a["question_id"], glob_turn) for a in test_annotations]
    out["fixed_absolute_time"] = score_l4(test_annotations, fixed_preds)
    out["fixed_absolute_time"]["alpha_global"] = alpha_glob

    # Category Temporal Prior — α per cat
    cat_preds = []
    for a in test_annotations:
        a_cat = alpha_per.get(a["subcategory"], alpha_glob)
        turn = int(round(a_cat)) if a_cat > 0 else 0
        cat_preds.append(_make_pred_for(a["question_id"], turn))
    out["category_temporal_prior"] = score_l4(test_annotations, cat_preds)
    out["category_temporal_prior"]["alpha_per_cat"] = alpha_per

    # Positive-oracle — only fire on positives (diagnostic only)
    oracle_preds = []
    for a in test_annotations:
        if a.get("is_positive"):
            a_cat = alpha_per.get(a["subcategory"], alpha_glob)
            oracle_preds.append(_make_pred_for(a["question_id"], int(round(a_cat))))
        else:
            oracle_preds.append(_make_pred_for(a["question_id"], None))
    out["positive_oracle_temporal_prior"] = score_l4(test_annotations, oracle_preds)

    return out


__all__ = ["shortcut_baselines"]
