"""L4 AlertScore: first-decision Hit and negative-session silence, aggregated over Core S1-S5."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from drivestreambench.data.io import read_jsonl, write_json
from drivestreambench.data.schemas import (
    L4_CORE_SUBCATEGORIES,
)
from drivestreambench.metrics._input_validation import require_prediction_coverage
from drivestreambench.protocol.frame_sampler import source_frame_step
from drivestreambench.protocol.parser import (
    DEFAULT_PARSER,
    ParserMode,
    extract_first_decision,
    has_any_non_silent_turn,
    is_literal_alert,
    window_for,
)

# Default τ values evaluated by the benchmark.
DEFAULT_TAUS: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
PRIMARY_TAU: float = 2.0

# Subcategory → eval mode (paper §3.4.1).
CAT_MODE: Dict[str, str] = {
    "lead_vehicle_start": "event_triggered",
    "lead_vehicle_brake": "event_triggered",
    "traffic_light_change": "event_triggered",
    "sudden_appearance": "event_triggered",
    "adjacent_cut_in": "target_timing",
    "vru_crossing": "target_timing",
}


# ---------------------------------------------------------------------------
# Per-session classification (the unit AlertScore is computed over)
# ---------------------------------------------------------------------------


@dataclass
class SessionOutcome:
    question_id: str
    is_positive: bool
    t_ref: Optional[float]
    alert_mode: str
    subcategory: str
    # parsed from prediction
    t_alert: Optional[float] = None
    first_reply: Optional[str] = None
    content_correct: bool = False  # = a_i in paper notation (first-decision is ALERT)
    had_any_valid_alert: bool = False  # NSR break: any non-silent response
    # parser mode actually used to score this session (recorded for audit)
    parser_mode: ParserMode = DEFAULT_PARSER


def reference_time(
    annotation: Dict[str, Any],
    *,
    fps: int = 2,
    source_fps: float = 10.0,
) -> Optional[float]:
    """Resolve the FPS-specific scoring reference in seconds.

    An explicit scalar ``t_ref`` is already in scoring coordinates and is
    used unchanged. Source-frame annotations map to the first left-aligned
    sampled frame at or after the reference, clamped to the final sample.
    """
    if fps not in (1, 2):
        raise ValueError("Scoring FPS must be 1 or 2")
    scalar = annotation.get("t_ref")
    if scalar is not None:
        try:
            value = float(scalar)
        except (TypeError, ValueError) as exc:
            raise ValueError("t_ref must be a finite time in seconds") from exc
        if not math.isfinite(value):
            raise ValueError("t_ref must be a finite time in seconds")
        return value
    anchor = annotation.get("t_ref_frame")
    if anchor is None:
        return None
    images = annotation.get("image_samples")
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError("Source-frame references require nonempty image_samples")
    try:
        frame = float(anchor)
    except (TypeError, ValueError) as exc:
        raise ValueError("t_ref_frame must be an in-bounds source-frame index") from exc
    if (
        isinstance(anchor, bool)
        or not math.isfinite(frame)
        or not frame.is_integer()
        or not 0 <= frame < len(images)
    ):
        raise ValueError("t_ref_frame must be an in-bounds source-frame index")
    step = source_frame_step(fps=fps, source_fps=source_fps)
    n_sampled = len(range(0, len(images), step))
    mapped_anchor = min(math.ceil(frame / step), n_sampled - 1)
    return mapped_anchor / fps


def build_outcomes(
    annotations: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    *,
    source_fps: float = 10.0,
    parser_mode: ParserMode = DEFAULT_PARSER,
    fps: int = 2,
) -> List[SessionOutcome]:
    """Pair annotation records with prediction records by ``question_id``.

    Annotation records use the release schema fields (see
    :class:`drivestreambench.data.schemas.Sample`). Required keys for L4
    scoring:

      ``question_id`` (str), ``question_type`` (str), ``alert_mode`` (str),
      ``sample_type`` (``"positive"`` / ``"negative"`` / null),
      ``t_ref_frame`` (int / null), ``image_samples`` (source frame paths).

    Source-frame references use the same FPS-specific grid as model inputs
    (2 FPS by default; pass ``fps=1`` for a native 1-FPS run). Explicit scalar
    ``t_ref`` values are already scoring times and are never re-quantized.

    The alternate annotation keys
    ``subcategory`` / ``is_positive`` / ``t_ref`` are also
    accepted when the release-schema keys are missing.

    Prediction record schema (DSB extension to release output_schema)::

      {
        "question_id": str,
        "model_name": str,
        "turns": [{"turn_idx": int, "video_time": float, "reply": str, ...}, ...]
      }

    Alternate ``model_response_list`` shape is also accepted (each turn
    dict carries ``role``, ``content``, ``time``).
    """
    require_prediction_coverage(annotations, predictions)
    pred_by_qid = {p["question_id"]: p for p in predictions}
    outcomes: List[SessionOutcome] = []
    for ann in annotations:
        qid = ann["question_id"]
        pred = pred_by_qid.get(qid)
        if pred is None:
            t_alert, first_reply = None, None
            had_any_alert = False
        else:
            turns = _normalize_turns(pred)
            t_alert, first_reply = extract_first_decision(
                turns, parser_mode=parser_mode
            )
            # NSR-side: under both parser modes, any non-silent turn that classifies
            # to ALERT or INVALID breaks negative silence (paper §3.4.2). Strict
            # parser distinguishes ALERT vs INVALID for the Hit side, but on the
            # NSR side both break silence. action_level has no INVALID class so any
            # non-silent turn is an ALERT action by definition.
            had_any_alert = has_any_non_silent_turn(turns)

        # Subcategory: prefer release-schema 'question_type', else fallback.
        subcat = str(ann.get("question_type") or ann.get("subcategory") or "")
        # is_positive: prefer release-schema 'sample_type', else fallback.
        if "sample_type" in ann:
            is_pos = ann.get("sample_type") == "positive"
        else:
            is_pos = bool(ann.get("is_positive", False))
        t_ref = reference_time(ann, fps=fps, source_fps=source_fps)

        # Hit-side a_i classification:
        #   strict_literal: first_reply must be literal "ALERT"
        #   action_level: any non-silent first_reply already counts (no INVALID)
        if first_reply is None:
            content_correct = False
        elif parser_mode == "action_level":
            content_correct = True
        else:
            content_correct = is_literal_alert(first_reply)

        outcomes.append(
            SessionOutcome(
                question_id=qid,
                is_positive=is_pos,
                t_ref=t_ref,
                alert_mode=str(
                    ann.get("alert_mode") or CAT_MODE.get(subcat, "event_triggered")
                ),
                subcategory=subcat,
                t_alert=t_alert,
                first_reply=first_reply,
                content_correct=content_correct,
                had_any_valid_alert=had_any_alert,
                parser_mode=parser_mode,
            )
        )
    return outcomes


def _normalize_turns(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accept both the new ``turns`` schema and the alternate ``model_response_list``."""
    if "turns" in pred:
        # convert {turn_idx, video_time, reply} → {role, time, content} (parser-friendly)
        return [
            {
                "role": "assistant",
                "time": t.get("video_time"),
                "content": t.get("reply", ""),
            }
            for t in pred["turns"]
        ]
    return list(pred.get("model_response_list", []))


# ---------------------------------------------------------------------------
# Component metrics
# ---------------------------------------------------------------------------


def compute_hit_at_tau(
    outcomes: Iterable[SessionOutcome], tau: float
) -> Tuple[float, int, int]:
    """Hit@τ = (# positives whose first ALERT lies in W_τ) / (# positives).

    Returns ``(hit_rate, hits, n_positives)``.
    """
    positives = [o for o in outcomes if o.is_positive]
    if not positives:
        return 0.0, 0, 0
    hits = 0
    for o in positives:
        if o.t_alert is None or o.t_ref is None or not o.content_correct:
            continue
        lo, hi = window_for(o.t_ref, tau, o.alert_mode)
        if lo <= o.t_alert <= hi:
            hits += 1
    return hits / len(positives), hits, len(positives)


def compute_nsr(outcomes: Iterable[SessionOutcome]) -> Tuple[float, int, int]:
    """NSR = (# negative sessions that remain silent) / (# negative sessions).

    Implements paper §3.4.2 b_j (denominator is total #negatives, not #alerted).
    Returns ``(nsr, n_silent, n_negatives)``.
    """
    negatives = [o for o in outcomes if not o.is_positive]
    if not negatives:
        return 1.0, 0, 0  # by convention; no negatives = perfect restraint
    n_silent = sum(1 for o in negatives if not o.had_any_valid_alert)
    return n_silent / len(negatives), n_silent, len(negatives)


def alert_score(hit_rate: float, nsr: float) -> float:
    """Harmonic mean of Hit@τ and NSR (= 1 − FA)."""
    if hit_rate <= 0.0 or nsr <= 0.0:
        return 0.0
    return 2.0 * hit_rate * nsr / (hit_rate + nsr)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def per_subcategory(
    outcomes: Sequence[SessionOutcome],
    taus: Sequence[float] = DEFAULT_TAUS,
) -> Dict[str, Dict[str, Any]]:
    """Compute Hit@τ / NSR / AlertScore@τ per subcategory."""
    by_cat: Dict[str, List[SessionOutcome]] = defaultdict(list)
    for o in outcomes:
        by_cat[o.subcategory].append(o)
    out: Dict[str, Dict[str, Any]] = {}
    for cat, cat_outcomes in by_cat.items():
        nsr, n_silent, n_neg = compute_nsr(cat_outcomes)
        cat_block: Dict[str, Any] = {
            "n_total": len(cat_outcomes),
            "n_pos": sum(1 for o in cat_outcomes if o.is_positive),
            "n_neg": n_neg,
            "nsr": round(nsr, 6),
            "fa_rate": round(1.0 - nsr, 6),
            "n_silent_neg": n_silent,
            "by_tau": {},
        }
        for tau in taus:
            hit, hits, npos = compute_hit_at_tau(cat_outcomes, tau)
            score = alert_score(hit, nsr)
            cat_block["by_tau"][str(tau)] = {
                "hit_rate": round(hit, 6),
                "hits": hits,
                "n_positives": npos,
                "alert_score": score,
            }
        out[cat] = cat_block
    return out


def core_macro(
    per_cat: Dict[str, Dict[str, Any]],
    taus: Sequence[float] = DEFAULT_TAUS,
    core_subcats: Sequence[str] = L4_CORE_SUBCATEGORIES,
) -> Dict[str, Any]:
    """Macro AlertScore@τ across the 5 L4-Core subcategories (paper §3.4.2).

    ``S6 sudden_appearance`` is intentionally **excluded**. Use
    :func:`per_subcategory` for the per-cat breakdown including S6.
    """
    cats_present = [c for c in core_subcats if c in per_cat]
    out: Dict[str, Any] = {
        "cats": cats_present,
        "n_cats": len(cats_present),
        "by_tau": {},
    }
    if not cats_present:
        return out
    # Macro NSR = mean of per-cat NSR
    macro_nsr = sum(per_cat[c]["nsr"] for c in cats_present) / len(cats_present)
    out["macro_nsr"] = round(macro_nsr, 6)
    out["macro_fa_rate"] = round(1.0 - macro_nsr, 6)
    for tau in taus:
        hits = [per_cat[c]["by_tau"][str(tau)]["hit_rate"] for c in cats_present]
        scores = [per_cat[c]["by_tau"][str(tau)]["alert_score"] for c in cats_present]
        out["by_tau"][str(tau)] = {
            "macro_hit": round(sum(hits) / len(hits), 6),
            "core_macro": sum(scores)
            / len(scores),  # retain precision before aggregation
        }
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def score_l4(
    annotations: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    taus: Sequence[float] = DEFAULT_TAUS,
    parser_mode: ParserMode = DEFAULT_PARSER,
    *,
    fps: int = 2,
    source_fps: float = 10.0,
) -> Dict[str, Any]:
    """High-level scorer.

    Parameters
    ----------
    annotations
        List of annotation dicts (see :class:`L4Sample`).
    predictions
        List of prediction dicts (see :class:`L4Prediction`).
    taus
        τ values to compute. Default is the paper's primary set (1, 2, 3, 5).
    fps
        Model input rate (2 FPS by default; 1 for native 1-FPS runs). Source
        frame references map to this sampled grid; scalar seconds remain intact.
    parser_mode
        ``"strict_literal"`` (default; instruction-following VLMs in
        the paper) or ``"action_level"`` (third-party streaming
        transfer baselines). See
        :mod:`drivestreambench.protocol.parser` for the full policy.

    Returns
    -------
    A nested dict::

        {
          "n_total": int,
          "n_pos": int,
          "n_neg": int,
          "parser_mode": str,
          "per_subcategory": {<cat>: {...}},
          "core_macro": {...},
          "by_tau_summary": {...},
          "primary_metric": {"name": "Core Macro AlertScore@2s", "value": float},
        }
    """
    outcomes = build_outcomes(
        annotations,
        predictions,
        parser_mode=parser_mode,
        fps=fps,
        source_fps=source_fps,
    )
    pcat = per_subcategory(outcomes, taus=taus)
    cm = core_macro(pcat, taus=taus)
    primary_value = (
        cm.get("by_tau", {}).get(str(PRIMARY_TAU), {}).get("core_macro", 0.0)
    )
    return {
        "n_total": len(outcomes),
        "n_pos": sum(1 for o in outcomes if o.is_positive),
        "n_neg": sum(1 for o in outcomes if not o.is_positive),
        "taus": list(taus),
        "parser_mode": parser_mode,
        "fps": fps,
        "source_fps": source_fps,
        "per_subcategory": pcat,
        "core_macro": cm,
        "primary_metric": {
            "name": f"Core Macro AlertScore@{PRIMARY_TAU}s",
            "value": primary_value,
        },
    }


def score_l4_files(
    annotation_path: str,
    prediction_path: str,
    output_path: Optional[str] = None,
    taus: Sequence[float] = DEFAULT_TAUS,
    parser_mode: ParserMode = DEFAULT_PARSER,
    *,
    fps: int = 2,
    source_fps: float = 10.0,
) -> Dict[str, Any]:
    """File-IO convenience wrapper around :func:`score_l4`.

    Used by ``scripts/score_predictions.sh`` and ``scripts/evaluate.py``.
    """
    annotations = read_jsonl(annotation_path)
    predictions = read_jsonl(prediction_path)
    result = score_l4(
        annotations,
        predictions,
        taus=taus,
        parser_mode=parser_mode,
        fps=fps,
        source_fps=source_fps,
    )
    result["annotation_file"] = annotation_path
    result["prediction_file"] = prediction_path
    if output_path:
        write_json(output_path, result)
    return result


__all__ = [
    "DEFAULT_TAUS",
    "PRIMARY_TAU",
    "CAT_MODE",
    "SessionOutcome",
    "build_outcomes",
    "reference_time",
    "compute_hit_at_tau",
    "compute_nsr",
    "alert_score",
    "per_subcategory",
    "core_macro",
    "score_l4",
    "score_l4_files",
]
