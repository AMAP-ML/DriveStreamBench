"""L3-gated AlertScore with video-level pairing of temporal QA and streaming sessions."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from drivestreambench.data.schemas import L4_CORE_SUBCATEGORIES
from drivestreambench.metrics.alertscore import (
    PRIMARY_TAU,
    SessionOutcome,
    alert_score,
    compute_hit_at_tau,
    compute_nsr,
)
from drivestreambench.metrics.l123_accuracy import normalize_reply
from drivestreambench.protocol.parser import window_for

# ── video_clip_id resolution ─────────────────────────────────────────

VIDEO_ID_NUPLAN_RE = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+_\d+_\d+)"
)
# Web-sourced S6 (sudden_appearance): scene_NNN_... or BV<hash>_...
VIDEO_ID_WEB_RE = re.compile(
    r"((?:scene_\d+|BV[a-zA-Z0-9]+)_sudden_appearance_test_[a-z]+_\d+)"
)
# nuPlan layout: /<root>/sensor_blobs/<set>/<VIDEO>/CAM_F0/<frame>.jpg
IMG_PATH_VIDEO_RE = re.compile(r"(?:^|/)([^/]+)/CAM_[^/]+/")
QID_LEVEL_SUFFIX_RE = re.compile(r"_(L[123])_(\d+)$")


def video_clip_id_from_qid(qid: str) -> Optional[str]:
    """Extract the video prefix from a qid (regex fallback path)."""
    qid = re.sub(r"^L[1-4]_", "", qid)
    m = VIDEO_ID_NUPLAN_RE.match(qid)
    if m:
        return m.group(1)
    m = VIDEO_ID_WEB_RE.match(qid)
    if m:
        return m.group(1)
    return None


def _video_id_from_image(image_path: str) -> Optional[str]:
    if not image_path:
        return None
    m = IMG_PATH_VIDEO_RE.search(image_path)
    return m.group(1) if m else None


def video_id_from_record(rec: Dict[str, Any]) -> Optional[str]:
    """Source-of-truth video_clip_id for an L1-L3 test record.

    Priority:
      1. ``image_samples[0]`` (or alternate ``images[0]``) path.
      2. qid prefix regex fallback.
    """
    imgs = rec.get("image_samples") or rec.get("images") or []
    if imgs:
        v = _video_id_from_image(imgs[0])
        if v:
            return v
    return video_clip_id_from_qid(rec.get("question_id", ""))


def _parse_qid_suffix(qid: str) -> Optional[Tuple[str, int]]:
    m = QID_LEVEL_SUFFIX_RE.search(qid)
    if not m:
        return None
    return m.group(1), int(m.group(2))


# ── L1-L3 probe correctness ───────────────────────────────────────────


def is_l13_correct(prediction: str, gold: str, is_multiple_choice: bool) -> bool:
    """Use the shared reactive parser; invalid and silent replies are wrong."""
    question_format = "mcq" if is_multiple_choice else "binary"
    normalized, status = normalize_reply(prediction, question_format)
    return status == "ok" and normalized == str(gold or "").strip().upper()


def _gold_answer(rec: Dict[str, Any]) -> str:
    """Extract the canonical short-form answer string from an L1-L3 record."""
    if "normalized_answer" in rec:
        return str(rec["normalized_answer"] or "")
    ans = rec.get("answer")
    if isinstance(ans, list) and ans:
        # release schema: answer = [{"content": "...", ...}]
        first = ans[0]
        if isinstance(first, dict):
            return str(first.get("content") or "")
    if isinstance(ans, str):
        return ans
    return ""


def _is_multiple_choice(rec: Dict[str, Any]) -> bool:
    if "is_multiple_choice" in rec:
        return bool(rec["is_multiple_choice"])
    answer = _gold_answer(rec).strip().upper()
    if answer in {"A", "B", "C", "D"}:
        return True
    if answer in {"YES", "NO"}:
        return False
    qtype = rec.get("question_type") or ""
    # Default heuristic: L2 is MC (A/B/C/D), L1 + L3 are YES/NO.
    return rec.get("question_level") == "L2" and qtype != ""


def build_video_canonical_correct(
    test_records: Sequence[Dict[str, Any]],
    pred_records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, bool]]:
    """For each ``video_id``, return ``{"L1": bool, "L2": bool, "L3": bool}``
    for the canonical (lowest-suffix) probe at each level.

    Identified predictions are matched by question_id. Legacy prediction
    exports without identifiers must retain the annotation row order.
    """
    if len(test_records) != len(pred_records):
        raise ValueError(
            f"L1-L3 test/pred length mismatch: "
            f"test={len(test_records)} pred={len(pred_records)}"
        )

    if any("question_id" in p for p in pred_records):
        from drivestreambench.metrics._input_validation import (
            require_prediction_coverage,
        )

        require_prediction_coverage(test_records, pred_records)
        pred_by_qid = {p["question_id"]: p for p in pred_records}
        pred_records = [pred_by_qid[t["question_id"]] for t in test_records]

    by_video_level: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    for i, t in enumerate(test_records):
        v = video_id_from_record(t)
        lvl = t.get("question_level")
        if not v or lvl not in {"L1", "L2", "L3"}:
            continue
        parsed = _parse_qid_suffix(t.get("question_id", ""))
        idx = parsed[1] if parsed else i
        by_video_level[(v, lvl)].append((idx, i))

    out: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for (v, lvl), items in by_video_level.items():
        items.sort()
        _, ti = items[0]
        gold_rec = test_records[ti]
        pred_rec = pred_records[ti]
        gold_str = _gold_answer(gold_rec)
        pred_str = (
            pred_rec.get("prediction")
            or pred_rec.get("raw_prediction")
            or pred_rec.get("output")
            or ""
        )
        out[v][lvl] = is_l13_correct(pred_str, gold_str, _is_multiple_choice(gold_rec))
    return out


# ── Core grounded scorer ──────────────────────────────────────────────


@dataclass
class L3GatedResult:
    tau: float
    gate_mode: str
    cats: List[str]
    n_pos_core: int
    n_pos_with_prerequisite: int
    coverage: float
    macro_hit_at_tau: float
    macro_grounded_hit_at_tau: float
    macro_nsr: float
    macro_alert_score_at_tau: float
    macro_grounded_alert_score_at_tau: float
    macro_temporal_grounding_gap_at_tau: float
    macro_l3_paired_acc: float
    per_cat: Dict[str, Dict[str, float]]


def score_l3_gated(
    l4_outcomes: Sequence[SessionOutcome],
    *,
    video_correct: Dict[str, Dict[str, bool]],
    tau: float = PRIMARY_TAU,
    gate_mode: str = "l3_only",
    core_subcats: Sequence[str] = L4_CORE_SUBCATEGORIES,
) -> L3GatedResult:
    """Compute Temporal-Grounded AlertScore@τ from pre-built L4 outcomes
    and a video → level → bool gate dictionary.

    Build outcomes with ``build_outcomes(..., fps=2)`` for the default protocol,
    or ``fps=1`` for a native 1-FPS run; the gate reuses those reference times.

    The L4 outcomes carry first-decision parsing already (from
    :func:`build_outcomes`) so this function reuses the canonical
    Hit / NSR machinery and only adds the gate.
    """
    if gate_mode not in {"l3_only", "all3_and"}:
        raise ValueError(f"gate_mode must be l3_only or all3_and, got {gate_mode!r}")

    # Bucket L4 outcomes by Core subcategory.
    by_cat: Dict[str, List[SessionOutcome]] = {c: [] for c in core_subcats}
    for o in l4_outcomes:
        if o.subcategory in by_cat:
            by_cat[o.subcategory].append(o)

    per_cat_out: Dict[str, Dict[str, float]] = {}
    macro_hit = macro_g_hit = macro_nsr_acc = 0.0
    macro_alert = macro_g_alert = macro_l3_acc = 0.0
    n_cat_with_data = 0
    total_pos = total_pos_with_pre = 0

    for cat in core_subcats:
        cat_outcomes = by_cat.get(cat, [])
        positives = [o for o in cat_outcomes if o.is_positive]
        if not positives:
            per_cat_out[cat] = {}
            continue
        n_cat_with_data += 1
        total_pos += len(positives)

        nsr, _, _ = compute_nsr(cat_outcomes)
        hit_rate, hits, _ = compute_hit_at_tau(cat_outcomes, tau)

        n_pos_with_pre = 0
        n_l3_correct = 0
        grounded_hits = 0
        for o in positives:
            vid = video_clip_id_from_qid(o.question_id)
            lvl = video_correct.get(vid, {}) if vid else {}
            if gate_mode == "l3_only":
                if "L3" not in lvl:
                    continue
                n_pos_with_pre += 1
                if lvl["L3"]:
                    n_l3_correct += 1
                # Hit eligibility for L4: same hit_at_tau machinery.
                if (
                    lvl["L3"]
                    and o.content_correct
                    and o.t_alert is not None
                    and o.t_ref is not None
                ):
                    lo, hi = window_for(o.t_ref, tau, o.alert_mode)
                    if lo <= o.t_alert <= hi:
                        grounded_hits += 1
            else:  # all3_and
                if not {"L1", "L2", "L3"}.issubset(lvl):
                    continue
                n_pos_with_pre += 1
                gate_ok = all(lvl[L] for L in ("L1", "L2", "L3"))
                if (
                    gate_ok
                    and o.content_correct
                    and o.t_alert is not None
                    and o.t_ref is not None
                ):
                    lo, hi = window_for(o.t_ref, tau, o.alert_mode)
                    if lo <= o.t_alert <= hi:
                        grounded_hits += 1

        total_pos_with_pre += n_pos_with_pre
        grounded_hit_rate = grounded_hits / len(positives) if positives else 0.0
        ifa = nsr  # alias for clarity (NSR = 1 − FA)
        cat_alert = alert_score(hit_rate, ifa)
        cat_g_alert = alert_score(grounded_hit_rate, ifa)
        l3_acc = (n_l3_correct / n_pos_with_pre) if n_pos_with_pre else 0.0
        per_cat_out[cat] = {
            "n_pos": float(len(positives)),
            "n_pos_with_prerequisite": float(n_pos_with_pre),
            "coverage": (n_pos_with_pre / len(positives)) if positives else 0.0,
            "n_l3_correct": float(n_l3_correct),
            "l3_paired_acc": l3_acc,
            "hit_at_tau": hit_rate,
            "grounded_hit_at_tau": grounded_hit_rate,
            "nsr": ifa,
            "alert_score_at_tau": cat_alert,
            "grounded_alert_score_at_tau": cat_g_alert,
        }
        macro_hit += hit_rate
        macro_g_hit += grounded_hit_rate
        macro_nsr_acc += ifa
        macro_alert += cat_alert
        macro_g_alert += cat_g_alert
        macro_l3_acc += l3_acc

    if n_cat_with_data == 0:
        # Empty result; keep shape stable for downstream CSV writers.
        return L3GatedResult(
            tau=tau,
            gate_mode=gate_mode,
            cats=list(core_subcats),
            n_pos_core=0,
            n_pos_with_prerequisite=0,
            coverage=0.0,
            macro_hit_at_tau=0.0,
            macro_grounded_hit_at_tau=0.0,
            macro_nsr=0.0,
            macro_alert_score_at_tau=0.0,
            macro_grounded_alert_score_at_tau=0.0,
            macro_temporal_grounding_gap_at_tau=0.0,
            macro_l3_paired_acc=0.0,
            per_cat=per_cat_out,
        )

    macro_hit /= n_cat_with_data
    macro_g_hit /= n_cat_with_data
    macro_nsr_acc /= n_cat_with_data
    macro_alert /= n_cat_with_data
    macro_g_alert /= n_cat_with_data
    macro_l3_acc /= n_cat_with_data

    return L3GatedResult(
        tau=tau,
        gate_mode=gate_mode,
        cats=list(core_subcats),
        n_pos_core=total_pos,
        n_pos_with_prerequisite=total_pos_with_pre,
        coverage=(total_pos_with_pre / total_pos) if total_pos else 0.0,
        macro_hit_at_tau=macro_hit,
        macro_grounded_hit_at_tau=macro_g_hit,
        macro_nsr=macro_nsr_acc,
        macro_alert_score_at_tau=macro_alert,
        macro_grounded_alert_score_at_tau=macro_g_alert,
        macro_temporal_grounding_gap_at_tau=macro_alert - macro_g_alert,
        macro_l3_paired_acc=macro_l3_acc,
        per_cat=per_cat_out,
    )


__all__ = [
    "L3GatedResult",
    "build_video_canonical_correct",
    "is_l13_correct",
    "score_l3_gated",
    "video_clip_id_from_qid",
    "video_id_from_record",
]
