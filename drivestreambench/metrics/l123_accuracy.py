"""L1/L2/L3 reactive QA accuracy.

L1 / L2 are single-frame VQA (binary YES/NO or 4-choice MCQ).
L3 uses a temporal frame sequence and the same reactive answer space.
``"NO REPLY"`` is reported separately and counts as incorrect. The scorer
provides:

* per-level accuracy
* per-question-type (binary / mcq) accuracy
* macro-F1 across the answer classes (binary YES/NO ; mcq A/B/C/D)

The scorer is robust to common formatting noise: leading/trailing
whitespace, punctuation, ``<think>`` tags, special tokens like
``<|im_end|>``, and case mismatches. Any non-normalisable response
is bucketed as ``invalid``; a literal ``"NO REPLY"`` is bucketed as
``no_reply`` (counts as "wrong" for accuracy but is reported
separately).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from drivestreambench.data.io import read_jsonl, write_json


NO_REPLY = "NO REPLY"

_SPECIAL_TOKEN_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>")
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_YESNO_PREFIX_RE = re.compile(r"^\s*(YES|NO)\b", re.IGNORECASE)
_OPTION_PREFIX_RE = re.compile(r"^\s*([A-D])(?:\b|[\.\)])", re.IGNORECASE)


def _clean(raw: str) -> str:
    s = _THINK_TAG_RE.sub("", raw or "")
    s = _SPECIAL_TOKEN_RE.sub(" ", s)
    return s.strip()


def normalize_reply(
    reply: Optional[str], question_format: str
) -> Tuple[Optional[str], str]:
    """Return ``(normalized, status)``.

    ``status`` ∈ ``{"ok", "no_reply", "invalid", "missing"}``.

    For ``binary`` question_format, normalised values are ``"YES"`` / ``"NO"``.
    For ``mcq``, normalised values are ``"A"`` / ``"B"`` / ``"C"`` / ``"D"``.
    """
    if reply is None:
        return None, "missing"
    cleaned = _clean(str(reply))
    if not cleaned:
        return None, "missing"
    upper = cleaned.upper().strip(" .,:;!?\"'")
    if upper == NO_REPLY:
        return NO_REPLY, "no_reply"

    if question_format == "binary":
        if upper in ("YES", "NO"):
            return upper, "ok"
        m = _YESNO_PREFIX_RE.match(cleaned)
        if m:
            return m.group(1).upper(), "ok"
        return None, "invalid"

    if question_format == "mcq":
        if upper in ("A", "B", "C", "D"):
            return upper, "ok"
        m = _OPTION_PREFIX_RE.match(cleaned)
        if m:
            return m.group(1).upper(), "ok"
        return None, "invalid"

    return None, "invalid"


def _per_class_prf(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision/recall/F1 from a confusion matrix.

    ``counts`` is keyed as ``counts[gold][pred] = count``.
    """
    classes = sorted(
        ({c for c in counts.keys()} | {p for d in counts.values() for p in d.keys()})
        - {"__INVALID__"}
    )
    out: Dict[str, Dict[str, float]] = {}
    for cls in classes:
        tp = counts.get(cls, {}).get(cls, 0)
        pred_total = sum(counts.get(g, {}).get(cls, 0) for g in counts.keys())
        gold_total = sum(counts.get(cls, {}).values())
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        out[cls] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": tp,
            "pred_total": pred_total,
            "gold_total": gold_total,
        }
    return out


def score_l123(
    annotations: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score reactive L1/L2/L3 predictions.

    Release annotations contain ``question_id``, ``question_level`` and
    ``answer``. Prediction records contain ``question_id`` and ``prediction``.
    Alternate ``level`` / ``question_format`` / ``reply`` keys are accepted.
    """
    pred_by_qid = {p["question_id"]: p for p in predictions}

    summary = {
        "total": 0,
        "correct": 0,
        "missing": 0,
        "invalid": 0,
        "no_reply": 0,
    }
    by_level: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "binary": {"total": 0, "correct": 0},
            "mcq": {"total": 0, "correct": 0},
            "no_reply": 0,
        }
    )
    by_qtype_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        "binary": defaultdict(lambda: defaultdict(int)),
        "mcq": defaultdict(lambda: defaultdict(int)),
    }

    for ann in annotations:
        qid = ann["question_id"]
        # Release schema: 'question_level'; alternate in-memory: 'level'.
        level = ann.get("question_level") or ann.get("level") or ""
        # question_format may be explicit (alternate) or inferred from gold answer.
        qf = ann.get("question_format")
        gold_raw = ann.get("answer")
        if qf is None:
            if gold_raw is None:
                qf = "unknown"
            elif str(gold_raw).strip().upper() in ("YES", "NO"):
                qf = "binary"
            elif str(gold_raw).strip().upper() in ("A", "B", "C", "D"):
                qf = "mcq"
            else:
                qf = "unknown"
        gold = str(gold_raw or "").strip().upper()
        pred = pred_by_qid.get(qid)
        # Release output_schema uses 'prediction'; alternate uses 'reply'.
        raw = None
        if pred is not None:
            raw = (
                pred.get("prediction")
                if pred.get("prediction") is not None
                else pred.get("reply")
            )
        norm, status = normalize_reply(raw, qf)

        summary["total"] += 1
        by_level[level]["total"] += 1
        if qf in ("binary", "mcq"):
            by_level[level][qf]["total"] += 1

        is_correct = status == "ok" and norm == gold
        if is_correct:
            summary["correct"] += 1
            by_level[level]["correct"] += 1
            if qf in ("binary", "mcq"):
                by_level[level][qf]["correct"] += 1
        if status == "missing":
            summary["missing"] += 1
        elif status == "invalid":
            summary["invalid"] += 1
        elif status == "no_reply":
            summary["no_reply"] += 1
            by_level[level]["no_reply"] += 1

        if qf in ("binary", "mcq"):
            predicted_class = norm if status == "ok" else "__INVALID__"
            by_qtype_counts[qf][gold][predicted_class] += 1

    summary["accuracy"] = (
        summary["correct"] / summary["total"] if summary["total"] else 0.0
    )

    by_level_out: Dict[str, Dict[str, Any]] = {}
    for level, blk in by_level.items():
        blk["accuracy"] = blk["correct"] / blk["total"] if blk["total"] else 0.0
        for qf in ("binary", "mcq"):
            qb = blk[qf]
            qb["accuracy"] = qb["correct"] / qb["total"] if qb["total"] else 0.0
        by_level_out[level] = blk

    by_qtype_out: Dict[str, Dict[str, Any]] = {}
    for qf, counts in by_qtype_counts.items():
        per_class = _per_class_prf(counts)
        macro_f1 = (
            (sum(c["f1"] for c in per_class.values()) / len(per_class))
            if per_class
            else 0.0
        )
        by_qtype_out[qf] = {
            "per_class": per_class,
            "macro_f1": round(macro_f1, 6),
            "n_scored": sum(c["gold_total"] for c in per_class.values()),
        }

    macro_l13 = (
        by_level_out.get("L1", {}).get("accuracy", 0.0)
        + by_level_out.get("L2", {}).get("accuracy", 0.0)
        + by_level_out.get("L3", {}).get("accuracy", 0.0)
    ) / 3.0

    return {
        "summary": {**summary, "macro_l1_l3_accuracy": round(macro_l13, 6)},
        "by_question_level": by_level_out,
        "by_question_type": by_qtype_out,
    }


def score_l123_files(
    annotation_path: str,
    prediction_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    annotations = read_jsonl(annotation_path)
    predictions = read_jsonl(prediction_path)
    result = score_l123(annotations, predictions)
    result["annotation_file"] = annotation_path
    result["prediction_file"] = prediction_path
    if output_path:
        write_json(output_path, result)
    return result


__all__ = [
    "normalize_reply",
    "score_l123",
    "score_l123_files",
]
