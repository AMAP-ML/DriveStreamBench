"""Build the 15-second L4 subset, retaining each positive reference frame and cropping negatives from the beginning."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from drivestreambench.data.io import read_jsonl, write_jsonl
from drivestreambench.data.schemas import SOURCE_FPS_DEFAULT


def _crop_positive(
    record: Dict[str, Any],
    *,
    target_n_frames: int,
) -> Dict[str, Any]:
    """Crop a positive session to ``target_n_frames`` frames around t_ref_frame.

    The window is centered on ``t_ref_frame`` when possible, then shifted
    to stay inside ``[0, n_frames)``. The new ``t_ref_frame`` is the
    original index minus the new window's start, so that the event still
    aligns with the same content frame in the cropped clip.
    """
    images: List[str] = list(record.get("image_samples") or [])
    n_frames = len(images)
    t_ref_orig = record.get("t_ref_frame")

    if n_frames <= target_n_frames or t_ref_orig is None:
        # Nothing to do — session already short enough or unannotated.
        record.setdefault("metadata", {})
        record["metadata"]["crop"] = {
            "kind": "positive_passthrough",
            "n_frames_in": n_frames,
            "n_frames_out": n_frames,
            "target_n_frames": target_n_frames,
            "t_ref_frame_in": t_ref_orig,
            "t_ref_frame_out": t_ref_orig,
            "window_start": 0,
            "window_end": n_frames,
        }
        return record

    half = target_n_frames // 2
    # Try to centre on t_ref_orig
    win_start = int(t_ref_orig) - half
    win_end = win_start + target_n_frames
    # Shift inside [0, n_frames]
    if win_start < 0:
        win_end -= win_start
        win_start = 0
    if win_end > n_frames:
        win_start -= (win_end - n_frames)
        win_end = n_frames
        if win_start < 0:
            win_start = 0
    new_images = images[win_start:win_end]
    new_t_ref = int(t_ref_orig) - win_start

    record = dict(record)
    record["image_samples"] = new_images
    record["t_ref_frame"] = new_t_ref
    meta = dict(record.get("metadata") or {})
    meta["crop"] = {
        "kind": "positive_window",
        "n_frames_in": n_frames,
        "n_frames_out": len(new_images),
        "target_n_frames": target_n_frames,
        "t_ref_frame_in": t_ref_orig,
        "t_ref_frame_out": new_t_ref,
        "window_start": win_start,
        "window_end": win_end,
    }
    record["metadata"] = meta
    return record


def _crop_negative(
    record: Dict[str, Any],
    *,
    target_n_frames: int,
) -> Dict[str, Any]:
    """Crop a negative session to its first ``target_n_frames`` frames."""
    images: List[str] = list(record.get("image_samples") or [])
    n_frames = len(images)
    new_images = images[:target_n_frames]
    record = dict(record)
    record["image_samples"] = new_images
    meta = dict(record.get("metadata") or {})
    meta["crop"] = {
        "kind": "negative_first_n",
        "n_frames_in": n_frames,
        "n_frames_out": len(new_images),
        "target_n_frames": target_n_frames,
        "t_ref_frame_in": None,
        "t_ref_frame_out": None,
        "window_start": 0,
        "window_end": min(n_frames, target_n_frames),
    }
    record["metadata"] = meta
    return record


def crop_record(
    record: Dict[str, Any],
    *,
    target_n_frames: int,
) -> Dict[str, Any]:
    """Dispatch to ``_crop_positive`` / ``_crop_negative`` by sample_type.

    Records that are not L4 (``question_level != "L4"``) or not annotated
    as positive/negative are returned unchanged with a passthrough audit
    note — the short subset is meaningful only for L4.
    """
    if record.get("question_level") != "L4":
        record = dict(record)
        meta = dict(record.get("metadata") or {})
        meta.setdefault("crop", {"kind": "non_L4_passthrough"})
        record["metadata"] = meta
        return record
    sample_type = record.get("sample_type")
    if sample_type == "positive":
        return _crop_positive(record, target_n_frames=target_n_frames)
    if sample_type == "negative":
        return _crop_negative(record, target_n_frames=target_n_frames)
    # Unlabelled — passthrough.
    record = dict(record)
    meta = dict(record.get("metadata") or {})
    meta.setdefault("crop", {"kind": "unlabelled_passthrough"})
    record["metadata"] = meta
    return record


def build_short(
    *,
    input_path: Path,
    output_path: Path,
    seconds: float = 15.0,
    source_fps: float = SOURCE_FPS_DEFAULT,
) -> Tuple[int, int]:
    """Read the input L4 annotation JSONL, write the short subset.

    Returns ``(n_records, n_cropped)``.
    """
    target_n_frames = int(round(seconds * source_fps))
    records = list(read_jsonl(input_path))
    out = []
    n_cropped = 0
    for r in records:
        new = crop_record(r, target_n_frames=target_n_frames)
        out.append(new)
        meta_crop = (new.get("metadata") or {}).get("crop") or {}
        if meta_crop.get("n_frames_in") and meta_crop.get("n_frames_out"):
            if meta_crop["n_frames_out"] < meta_crop["n_frames_in"]:
                n_cropped += 1
    write_jsonl(output_path, out)
    print(
        f"[build_short] {input_path} → {output_path}\n"
        f"  records:        {len(out)}\n"
        f"  cropped:        {n_cropped}\n"
        f"  target frames:  {target_n_frames} (= {seconds}s @ {source_fps} FPS)",
        file=sys.stderr,
    )
    return len(out), n_cropped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="L4 annotation JSONL (e.g. annotations/l4_core_test.jsonl).")
    p.add_argument("--output", type=Path, required=True, help="Where to write the cropped (short) annotation JSONL.")
    p.add_argument("--seconds", type=float, default=15.0, help="Target window length in seconds (default 15).")
    p.add_argument("--source-fps", type=float, default=SOURCE_FPS_DEFAULT, help=f"Source FPS (default {SOURCE_FPS_DEFAULT}).")
    args = p.parse_args()

    if not args.input.is_file():
        sys.exit(f"input not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_short(
        input_path=args.input,
        output_path=args.output,
        seconds=args.seconds,
        source_fps=args.source_fps,
    )


if __name__ == "__main__":
    main()
