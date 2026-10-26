"""Build multi-turn L4 SFT records in the standard ms-swift format.

User messages contain ordered <image> placeholders, with matching paths in
the images list. Each assistant message carries its loss-inclusion flag.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from drivestreambench.data.io import iter_jsonl, write_jsonl
from drivestreambench.data.schemas import SOURCE_FPS_DEFAULT, Sample
from drivestreambench.protocol.frame_sampler import source_frame_step
from drivestreambench.protocol.prompts import L4_SYSTEM_PROMPT


def session_to_turns(
    sample: Sample,
    *,
    fps: int = 2,
    frames_per_turn: int = 2,
    decision_cadence_hz: float = 1.0,
    source_fps: float = SOURCE_FPS_DEFAULT,
) -> List[Dict[str, Any]]:
    """Build the ordered turn-list for one L4 session.

    Returns a list of dicts ``{role, content, video_time, frame_idxs}``
    suitable for downstream conversion into vendor-specific training
    formats.
    """
    if frames_per_turn < 1 or decision_cadence_hz <= 0:
        raise ValueError("frames_per_turn and decision_cadence_hz must be positive")
    step = source_frame_step(fps=fps, source_fps=source_fps)
    if not math.isclose(frames_per_turn / fps, 1 / decision_cadence_hz):
        raise ValueError("frames_per_turn / fps must match the decision interval")
    n_frames = len(sample.image_samples)
    if not n_frames:
        raise ValueError(f"Sample {sample.question_id!r} has no frames")
    anchor = sample.t_ref_frame if sample.is_positive else None
    if sample.is_positive and anchor is None:
        raise ValueError(
            f"Positive sample {sample.question_id!r} has no reference frame"
        )
    if anchor is not None and not 0 <= anchor < n_frames:
        raise ValueError(
            f"Reference frame for {sample.question_id!r} is outside the clip"
        )
    # Training retains the event anchor and the initial scene frame.
    origin = anchor if anchor is not None else 0
    keep = {0, origin}
    for sign in (-1, 1):
        idx = origin + sign * step
        while 0 <= idx < n_frames:
            keep.add(idx)
            idx += sign * step
    indices = sorted(keep)
    turns: List[Dict[str, Any]] = []
    for start in range(0, len(indices), frames_per_turn):
        group = indices[start : start + frames_per_turn]
        is_alert = anchor is not None and anchor in group
        turns.append(
            {
                "role": "assistant",
                "content": "ALERT" if is_alert else "NO REPLY",
                "video_time": (start + len(group)) / fps,
                "frame_idxs": group,
                "is_alert_turn": is_alert,
            }
        )
    return turns


def to_ms_swift_record(
    sample: Sample,
    *,
    fps: int = 2,
    frames_per_turn: int = 2,
    decision_cadence_hz: float = 1.0,
    source_fps: float = SOURCE_FPS_DEFAULT,
    loss_mask_rate: float = 0.5,
    rng: Optional[random.Random] = None,
    dataroot: Optional[str] = None,
) -> Dict[str, Any]:
    """Produce one standard ms-swift record."""
    if rng is None:
        rng = random.Random(0)
    turns = session_to_turns(
        sample,
        fps=fps,
        frames_per_turn=frames_per_turn,
        decision_cadence_hz=decision_cadence_hz,
        source_fps=source_fps,
    )
    messages: List[Dict[str, Any]] = [{"role": "system", "content": L4_SYSTEM_PROMPT}]
    images: List[str] = []
    for i, t in enumerate(turns):
        # User turn carries the new frames since last decision.
        user_content: List[Dict[str, Any]] = []
        for fidx in t["frame_idxs"]:
            if 0 <= fidx < len(sample.image_samples):
                rel = sample.image_samples[fidx]
                full = str(Path(dataroot) / rel) if dataroot is not None else rel
                images.append(full)
                user_content.append({"type": "image"})
        if i == 0:
            # First user turn carries the watch instruction.
            user_content.append({"type": "text", "text": sample.question})
        user_text = "\n".join(
            "<image>" if item["type"] == "image" else item["text"]
            for item in user_content
        )
        messages.append({"role": "user", "content": user_text})
        # Loss masking: drop a fraction of NO REPLY turns
        if t["content"] == "NO REPLY" and rng.random() < loss_mask_rate:
            include_loss = False
        else:
            include_loss = True
        messages.append(
            {"role": "assistant", "content": t["content"], "loss": include_loss}
        )

    return {
        "messages": messages,
        "images": images,
        "question_id": sample.question_id,
        "video_name": sample.video_name,
        "subcategory": sample.question_type,
        "is_positive": sample.is_positive,
    }


def build(
    *,
    annotations_path: str | Path,
    output_path: str | Path,
    fps: int = 2,
    frames_per_turn: int = 2,
    decision_cadence_hz: float = 1.0,
    source_fps: float = SOURCE_FPS_DEFAULT,
    loss_mask_rate: float = 0.5,
    seed: int = 42,
    progress: bool = True,
    dataroot: Optional[str] = None,
) -> int:
    """Convert all L4 samples in ``annotations_path`` to SFT records.

    If ``dataroot`` is provided, every entry of the per-record
    ``images`` list is rewritten to ``{dataroot}/{image_samples[i]}``;
    otherwise the relative paths from the annotation are kept verbatim
    (the training-time loader is then expected to resolve dataroot).
    """
    rng = random.Random(seed)
    out_records: List[Dict[str, Any]] = []
    n_seen = 0
    for raw in iter_jsonl(annotations_path):
        sample = Sample.from_dict(raw)
        if sample.question_level != "L4":
            continue
        n_seen += 1
        rec = to_ms_swift_record(
            sample,
            fps=fps,
            frames_per_turn=frames_per_turn,
            decision_cadence_hz=decision_cadence_hz,
            source_fps=source_fps,
            loss_mask_rate=loss_mask_rate,
            rng=rng,
            dataroot=dataroot,
        )
        out_records.append(rec)
        if progress and n_seen % 500 == 0:
            print(f"[build_sft_data] processed {n_seen} samples", flush=True)
    write_jsonl(output_path, out_records)
    if progress:
        print(
            f"[build_sft_data] wrote {len(out_records)} SFT records → {output_path}",
            flush=True,
        )
    return len(out_records)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build SFT training JSONL from L4 release annotations"
    )
    ap.add_argument(
        "--annotations",
        required=True,
        help="L4 release JSONL path (e.g. l4_core_train.jsonl)",
    )
    ap.add_argument("--output", required=True, help="output SFT JSONL path")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--frames-per-turn", type=int, default=2)
    ap.add_argument("--cadence", type=float, default=1.0)
    ap.add_argument("--source-fps", type=float, default=SOURCE_FPS_DEFAULT)
    ap.add_argument(
        "--loss-mask-rate",
        type=float,
        default=0.5,
        help="fraction of NO REPLY assistant turns to mask out of loss",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dataroot",
        default=None,
        help="optional prefix for image paths "
        "(rewrites images[i] to {dataroot}/{image_samples[i]})",
    )
    args = ap.parse_args()
    build(
        annotations_path=args.annotations,
        output_path=args.output,
        fps=args.fps,
        frames_per_turn=args.frames_per_turn,
        decision_cadence_hz=args.cadence,
        source_fps=args.source_fps,
        loss_mask_rate=args.loss_mask_rate,
        seed=args.seed,
        dataroot=args.dataroot,
    )


if __name__ == "__main__":
    main()
