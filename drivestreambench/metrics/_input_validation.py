"""Record-ID integrity checks for complete benchmark predictions."""

from collections import Counter


def require_prediction_coverage(annotations, predictions):
    """Require exactly one prediction for every annotation ID."""
    groups = {}
    for label, records in [("annotation", annotations), ("prediction", predictions)]:
        ids = [record.get("question_id") for record in records]
        if any(not isinstance(qid, str) or not qid for qid in ids):
            raise ValueError(f"Every {label} requires a nonempty question_id")
        duplicates = [qid for qid, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate {label} question_id: {duplicates[0]}")
        groups[label] = set(ids)
    missing = groups["annotation"] - groups["prediction"]
    extra = groups["prediction"] - groups["annotation"]
    if missing or extra:
        raise ValueError(
            "Prediction coverage mismatch: "
            f"{len(missing)} missing and {len(extra)} unexpected question_id(s)"
        )
