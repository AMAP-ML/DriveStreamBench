"""Build single-turn L1-L3 SFT records in the standard ms-swift format.

Binary and multiple-choice questions use their corresponding system prompts.
All annotated frames are retained unless an explicit max_frames cap is provided.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from drivestreambench.data.io import iter_jsonl, write_jsonl
from drivestreambench.data.schemas import Sample
from drivestreambench.protocol.prompts import (
    L1L2_MCQ_SYSTEM_PROMPT,
    L1L2L3_BINARY_SYSTEM_PROMPT,
    build_l1l2l3_user_message,
)


def _uniform_subsample(n_total: int, n_keep: int) -> List[int]:
    """Return ``n_keep`` evenly-spaced indices into ``[0, n_total)``.

    If ``n_total <= n_keep``, returns ``range(n_total)``.
    """
    if n_total <= n_keep:
        return list(range(n_total))
    step = (n_total - 1) / (n_keep - 1) if n_keep > 1 else 0.0
    return [min(n_total - 1, int(round(i * step))) for i in range(n_keep)]


def to_l13_record(
    sample: Sample,
    *,
    max_frames: Optional[int] = None,
    dataroot: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert one L1/L2/L3 release record into a single-turn SFT record.

    Returns ``None`` for L4 records or for samples missing a usable
    answer (defensive: skip rather than crash).
    """
    if sample.question_level not in ("L1", "L2", "L3"):
        return None
    answer = (sample.answer or "").strip().upper()
    if not answer:
        return None

    n = len(sample.image_samples)
    if n == 0:
        return None
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when specified")
    keep_idxs = (
        list(range(n)) if max_frames is None else _uniform_subsample(n, max_frames)
    )
    images: List[str] = []
    user_content: List[Dict[str, Any]] = []
    for idx in keep_idxs:
        rel = sample.image_samples[idx]
        full = str(Path(dataroot) / rel) if dataroot is not None else rel
        images.append(full)
        user_content.append({"type": "image"})
    user_content.append(
        {
            "type": "text",
            "text": build_l1l2l3_user_message(sample.question, options=None),
        }
    )

    system_prompt = (
        L1L2_MCQ_SYSTEM_PROMPT
        if sample.question_format == "mcq"
        else L1L2L3_BINARY_SYSTEM_PROMPT
    )
    user_text = "\n".join(
        "<image>" if item["type"] == "image" else item["text"] for item in user_content
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer, "loss": True},
    ]
    return {
        "messages": messages,
        "images": images,
        "question_id": sample.question_id,
        "video_name": sample.video_name,
        "level": sample.question_level,
    }


def build(
    *,
    annotations_paths: List[Path],
    output_path: Path,
    max_frames: Optional[int] = None,
    dataroot: Optional[str] = None,
    progress: bool = True,
) -> int:
    """Convert every L1/L2/L3 sample in ``annotations_paths`` to SFT records.

    Multiple input paths are merged in the order given (caller is
    responsible for shuffling if a randomised order is wanted).
    """
    out_records: List[Dict[str, Any]] = []
    n_seen_by_level: Dict[str, int] = {"L1": 0, "L2": 0, "L3": 0}
    n_skipped = 0
    for ap in annotations_paths:
        for raw in iter_jsonl(ap):
            sample = Sample.from_dict(raw)
            rec = to_l13_record(
                sample,
                max_frames=max_frames,
                dataroot=dataroot,
            )
            if rec is None:
                n_skipped += 1
                continue
            out_records.append(rec)
            n_seen_by_level[rec["level"]] = n_seen_by_level.get(rec["level"], 0) + 1
            if progress and len(out_records) % 1000 == 0:
                print(
                    f"[build_sft_data_l13] {len(out_records)} records so far",
                    flush=True,
                )
    write_jsonl(output_path, out_records)
    if progress:
        breakdown = " ".join(f"{lvl}={n}" for lvl, n in sorted(n_seen_by_level.items()))
        print(
            f"[build_sft_data_l13] wrote {len(out_records)} records → {output_path}  "
            f"({breakdown}; skipped={n_skipped})",
            flush=True,
        )
    return len(out_records)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build SFT JSONL for L1/L2/L3 reactive QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--annotations",
        required=True,
        nargs="+",
        help="one or more L1/L2/L3 release JSONL paths "
        "(typically annotations/{l1,l2,l3}_train.jsonl)",
    )
    ap.add_argument("--output", required=True, help="output SFT JSONL path")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="optional uniform frame cap; all annotated frames are retained by default",
    )
    ap.add_argument("--dataroot", default=None, help="optional prefix for image paths")
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="randomly shuffle output records (use with --seed for reproducibility)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    annotations_paths = [Path(p) for p in args.annotations]
    for p in annotations_paths:
        if not p.is_file():
            raise SystemExit(f"annotations file not found: {p}")

    n = build(
        annotations_paths=annotations_paths,
        output_path=Path(args.output),
        max_frames=args.max_frames,
        dataroot=args.dataroot,
    )

    if args.shuffle and n > 0:
        # In-place shuffle of the output JSONL
        records = list(iter_jsonl(args.output))
        rng = random.Random(args.seed)
        rng.shuffle(records)
        write_jsonl(args.output, records)
        print(
            f"[build_sft_data_l13] shuffled {len(records)} records (seed={args.seed})"
        )


if __name__ == "__main__":
    main()
