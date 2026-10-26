#!/usr/bin/env python3
"""Prepare native streaming-model inputs and matching FPS-specific reference times."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivestreambench.data.io import read_jsonl
from drivestreambench.protocol.prompts import L4_SYSTEM_PROMPT

SOURCE_FPS = 10


def prepare_records(
    annotations: Sequence[dict[str, Any]], *, dataroot: Path, fps: int,
    check_images: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build native input/gold and scorer gold without modifying source records.

    Frames use the left-aligned source grid. A positive reference maps to the
    first sampled frame at or after the source reference, clamped to the last
    sampled frame. Scorer gold stores this time directly in seconds.
    """
    if fps not in (1, 2):
        raise ValueError("Supported native input rates are 1 and 2 FPS")
    if not annotations:
        raise ValueError("Annotations must contain at least one L4 record")
    dataroot = Path(dataroot).expanduser().resolve()
    step = SOURCE_FPS // fps
    inputs, golds, scoring = [], [], []
    seen: set[str] = set()
    for row in annotations:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError(f"Missing or duplicate question_id: {qid!r}")
        seen.add(qid)
        if row.get("question_level") != "L4":
            raise ValueError(f"Expected an L4 annotation: {qid}")
        question = row.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError(f"Missing watch instruction: {qid}")
        images = row.get("image_samples")
        if not isinstance(images, list) or not images:
            raise ValueError(f"Missing source frames: {qid}")
        category = row.get("question_type")
        mode, polarity = row.get("alert_mode"), row.get("sample_type")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Missing question_type: {qid}")
        if mode not in ("event_triggered", "target_timing"):
            raise ValueError(f"Invalid alert_mode for {qid}: {mode!r}")
        if polarity not in ("positive", "negative"):
            raise ValueError(f"Invalid sample_type for {qid}: {polarity!r}")
        absolute = []
        for relative in images:
            if not isinstance(relative, str) or not relative or "\x00" in relative:
                raise ValueError(f"Invalid source image path for {qid}: {relative!r}")
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Image paths must be relative to dataroot: {relative}")
            absolute.append(str(dataroot.joinpath(*path.parts)))
        sampled = absolute[::step]
        if check_images:
            for path in sampled:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": L4_SYSTEM_PROMPT}
        ]
        for start in range(0, len(sampled), fps):
            content = [{"type": "image", "image": p} for p in sampled[start:start + fps]]
            if start == 0:
                content.append({"type": "text", "text": question})
            conversation.append({"role": "user", "content": content})
        inputs.append({"question_id": qid, "question_level": "L4", "conversation": conversation})
        t_ref = None
        if polarity == "positive":
            anchor = row.get("t_ref_frame")
            if isinstance(anchor, bool) or not isinstance(anchor, int) or not 0 <= anchor < len(images):
                raise ValueError(f"Positive source reference is out of bounds: {qid}")
            sampled_anchor = min((anchor + step - 1) // step, len(sampled) - 1)
            t_ref = sampled_anchor / fps
        golds.append({
            "question_id": qid, "question_level": "L4",
            "conversation": [{"role": "user", "content": question, "time": 0.5}],
            "answer": [{"content": "ALERT" if polarity == "positive" else "NO REPLY",
                        "t_ref": t_ref, "alert_mode": mode}],
        })
        scoring.append({
            "question_id": qid, "question_type": category,
            "sample_type": polarity, "alert_mode": mode, "t_ref": t_ref,
        })
    return inputs, golds, scoring


def prepare_files(*, annotations: Path, dataroot: Path, fps: int, output_dir: Path) -> int:
    """Write a new preparation directory after validating the complete input."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    inputs, golds, scoring = prepare_records(read_jsonl(annotations), dataroot=dataroot, fps=fps)
    output_dir.mkdir(parents=True)
    for name, rows in (("input.json", inputs), ("gold.json", golds)):
        (output_dir / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "scoring_annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scoring), encoding="utf-8"
    )
    return len(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Released 10-FPS L4 annotation JSONL")
    parser.add_argument("--dataroot", type=Path, required=True, help="Root for annotation-relative frame paths")
    parser.add_argument("--fps", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for input.json, gold.json and scoring_annotations.jsonl")
    args = parser.parse_args()
    count = prepare_files(annotations=args.annotations, dataroot=args.dataroot,
                          fps=args.fps, output_dir=args.output_dir)
    print(f"Prepared {count} L4 samples at {args.fps} FPS in {args.output_dir}")


if __name__ == "__main__":
    main()
