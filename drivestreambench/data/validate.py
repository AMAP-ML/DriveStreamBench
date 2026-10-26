"""Annotation validator — catches schema violations at data-prep time.

Aligned to the release schema in ``DriveStreamBench_release/schema.json``.
Run via:

    python -m drivestreambench.data.validate annotations/l4_core_test.jsonl
"""
from __future__ import annotations

import argparse
from typing import Iterable, List, Optional, Tuple

from drivestreambench.data.io import read_jsonl
from drivestreambench.data.schemas import (
    ALERT_MODES,
    ALL_QUESTION_TYPES,
    QUESTION_LEVELS,
    SAMPLE_TYPES,
    SOURCE_TYPES,
    SPLITS,
)


REQUIRED_FIELDS: Tuple[str, ...] = (
    "question_id", "question_level", "video_name", "question_type",
    "split", "sample_type", "alert_mode", "source_type",
    "image_samples", "t_ref_frame", "question", "answer",
    "trigger_candidate_count",
)
# `metadata` is intentionally optional. The published release schema




def validate(records: Iterable[dict]) -> List[Tuple[int, str]]:
    """Return list of (record_index, error_message). Empty list = OK."""
    errors: List[Tuple[int, str]] = []
    for i, r in enumerate(records):
        # Required fields present
        for f in REQUIRED_FIELDS:
            if f not in r:
                errors.append((i, f"missing field '{f}'"))

        # Enums
        level = r.get("question_level")
        if level is not None and level not in QUESTION_LEVELS:
            errors.append((i, f"unknown question_level '{level}'"))

        qtype = r.get("question_type")
        if qtype is not None and qtype not in ALL_QUESTION_TYPES:
            errors.append((i, f"unknown question_type '{qtype}'"))

        split = r.get("split")
        if split is not None and split not in SPLITS:
            errors.append((i, f"unknown split '{split}'"))

        sample_type = r.get("sample_type")
        if sample_type not in SAMPLE_TYPES:
            errors.append((i, f"unknown sample_type '{sample_type}'"))

        alert_mode = r.get("alert_mode")
        if alert_mode is not None and alert_mode not in ALERT_MODES:
            errors.append((i, f"unknown alert_mode '{alert_mode}'"))

        source_type = r.get("source_type")
        if source_type is not None and source_type not in SOURCE_TYPES:
            errors.append((i, f"unknown source_type '{source_type}'"))

        # Cross-field invariants
        if level == "L4":
            if alert_mode is None:
                errors.append((i, "L4 sample requires non-null alert_mode"))
            if sample_type == "positive" and r.get("t_ref_frame") is None:
                errors.append((i, "L4 positive sample requires non-null t_ref_frame"))
            if sample_type == "negative" and r.get("t_ref_frame") is not None:
                errors.append((i, "L4 negative sample should have null t_ref_frame"))

        # image_samples must be relative paths (no leading /)
        for j, p in enumerate(r.get("image_samples") or []):
            if not isinstance(p, str) or p.startswith("/"):
                errors.append((i, f"image_samples[{j}] must be relative path, got: {p!r}"))
                break

        # Metadata is optional in the public 13-field schema. A derived


    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate DSB annotation JSONL (release schema)")
    ap.add_argument("path", help="path to the annotation JSONL file")
    ap.add_argument("--max-errors", type=int, default=20,
                    help="max errors to print before truncating (default 20)")
    args = ap.parse_args()
    records = read_jsonl(args.path)
    errors = validate(records)
    if errors:
        print(f"FAIL: {len(errors)} validation errors in {args.path}")
        for i, msg in errors[:args.max_errors]:
            print(f"  record[{i}]: {msg}")
        if len(errors) > args.max_errors:
            print(f"  (... {len(errors) - args.max_errors} more)")
        raise SystemExit(1)
    print(f"OK: {len(records)} records in {args.path} pass release-schema validation")


if __name__ == "__main__":
    main()
