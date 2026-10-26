"""Binary event F1 for L3 temporal QA."""

from typing import Any, Dict, List

from drivestreambench.metrics.l123_accuracy import normalize_reply


def binary_f1_l3(
    annotations: List[Dict[str, Any]], predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute binary F1 for L3 event occurrence.

    Event labels: ``positive`` (sample_type) → event_occurred=1,
    ``negative`` → 0. The parsed prediction is mapped the same way
    (Yes→1 / No→0). Predictions that fail to normalise count as the
    wrong class (FN if gold=1, FP if gold=0).
    """
    pred_by_qid = {p["question_id"]: p for p in predictions}
    tp = fp = fn = tn = 0
    n_total = 0
    for a in annotations:
        # Take only L3 records.
        if a.get("question_level") != "L3":
            continue
        n_total += 1
        gold_pos = a.get("sample_type") == "positive"
        pred = pred_by_qid.get(a["question_id"])
        if pred is None:
            # missing prediction = wrong
            if gold_pos:
                fn += 1
            else:
                fp += 1
            continue
        norm, status = normalize_reply(pred.get("prediction"), "binary")
        if status != "ok":
            if gold_pos:
                fn += 1
            else:
                fp += 1
            continue
        pred_pos = norm == "YES"
        if gold_pos and pred_pos:
            tp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        elif (not gold_pos) and pred_pos:
            fp += 1
        else:
            tn += 1
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    return {
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_total": n_total,
    }
