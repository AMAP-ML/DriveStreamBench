"""Tests for :mod:`drivestreambench.data.build_short`.

The short-version subset is the closed-source-friendly evaluation
slice (paper §6). Cropping must:

  - keep all records (count preserved)
  - rewrite ``t_ref_frame`` for positives so it still points at the
    same content frame in the cropped clip
  - leave negatives' ``t_ref_frame`` (which is ``None``) alone
  - record the crop decision under ``metadata.crop`` for audit
  - be idempotent under "already short enough" inputs
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from drivestreambench.data.build_short import build_short, crop_record


def _ann(qid, *, sample_type, t_ref_frame, n_frames, level="L4", cat="lead_vehicle_start"):
    return {
        "question_id": qid,
        "question_level": level,
        "video_name": "v",
        "question_type": cat,
        "split": "test",
        "sample_type": sample_type,
        "alert_mode": "event_triggered",
        "source_type": "nuplan",
        "image_samples": [f"v/CAM_F0/{i:04d}.jpg" for i in range(n_frames)],
        "t_ref_frame": t_ref_frame,
        "question": "watch",
        "answer": None,
        "trigger_candidate_count": 1 if sample_type == "positive" else 0,
        "metadata": {"original_question_id": qid},
    }


def test_negative_keeps_first_n_frames():
    rec = _ann("n1", sample_type="negative", t_ref_frame=None, n_frames=300)
    out = crop_record(rec, target_n_frames=150)
    assert len(out["image_samples"]) == 150
    assert out["t_ref_frame"] is None
    crop = out["metadata"]["crop"]
    assert crop["kind"] == "negative_first_n"
    assert crop["n_frames_in"] == 300 and crop["n_frames_out"] == 150


def test_negative_passthrough_when_already_short():
    rec = _ann("n1", sample_type="negative", t_ref_frame=None, n_frames=80)
    out = crop_record(rec, target_n_frames=150)
    assert len(out["image_samples"]) == 80
    crop = out["metadata"]["crop"]
    assert crop["n_frames_in"] == 80 and crop["n_frames_out"] == 80


def test_positive_centered_window_remaps_t_ref():
    """t_ref_frame=200 in a 400-frame clip cropped to 150: the window
    should be centred at 200, so [125, 275). New t_ref = 200 - 125 = 75."""
    rec = _ann("p1", sample_type="positive", t_ref_frame=200, n_frames=400)
    out = crop_record(rec, target_n_frames=150)
    crop = out["metadata"]["crop"]
    assert crop["window_start"] == 125
    assert crop["window_end"] == 275
    assert out["t_ref_frame"] == 75
    assert len(out["image_samples"]) == 150
    # Audit field preserves the original t_ref.
    assert crop["t_ref_frame_in"] == 200
    assert crop["t_ref_frame_out"] == 75


def test_positive_t_ref_near_start_shifts_to_fit():
    """t_ref_frame=10 — window cannot centre, must shift to start at 0.
    New t_ref = 10."""
    rec = _ann("p1", sample_type="positive", t_ref_frame=10, n_frames=400)
    out = crop_record(rec, target_n_frames=150)
    crop = out["metadata"]["crop"]
    assert crop["window_start"] == 0
    assert crop["window_end"] == 150
    assert out["t_ref_frame"] == 10


def test_positive_t_ref_near_end_shifts_to_fit():
    """t_ref_frame=395 in 400-frame clip — window shifts back to end at 400.
    Window = [250, 400), new t_ref = 395 - 250 = 145."""
    rec = _ann("p1", sample_type="positive", t_ref_frame=395, n_frames=400)
    out = crop_record(rec, target_n_frames=150)
    crop = out["metadata"]["crop"]
    assert crop["window_start"] == 250
    assert crop["window_end"] == 400
    assert out["t_ref_frame"] == 145


def test_positive_passthrough_when_already_short():
    rec = _ann("p1", sample_type="positive", t_ref_frame=42, n_frames=120)
    out = crop_record(rec, target_n_frames=150)
    crop = out["metadata"]["crop"]
    assert crop["kind"] == "positive_passthrough"
    assert out["t_ref_frame"] == 42  # unchanged
    assert len(out["image_samples"]) == 120


def test_t_ref_alignment_invariant():
    """Critical invariant: the frame referenced by t_ref before and
    after cropping must be the same actual content frame (same image
    path). Any drift here breaks closed-source scoring (paper §6)."""
    rec = _ann("p1", sample_type="positive", t_ref_frame=200, n_frames=400)
    img_at_t_ref_before = rec["image_samples"][rec["t_ref_frame"]]
    out = crop_record(rec, target_n_frames=150)
    img_at_t_ref_after = out["image_samples"][out["t_ref_frame"]]
    assert img_at_t_ref_before == img_at_t_ref_after


def test_non_l4_passes_through():
    rec = _ann("l1q", sample_type=None, t_ref_frame=None, n_frames=50, level="L1")
    out = crop_record(rec, target_n_frames=150)
    crop = out["metadata"]["crop"]
    assert crop["kind"] == "non_L4_passthrough"
    assert len(out["image_samples"]) == 50


def test_build_short_file_io(tmp_path: Path):
    in_path = tmp_path / "l4_core_test.jsonl"
    out_path = tmp_path / "l4_short_test.jsonl"
    records = [
        _ann("p1", sample_type="positive", t_ref_frame=200, n_frames=400),
        _ann("n1", sample_type="negative", t_ref_frame=None, n_frames=300),
        _ann("p2", sample_type="positive", t_ref_frame=42, n_frames=120),  # passthrough
    ]
    in_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    n_total, n_cropped = build_short(input_path=in_path, output_path=out_path, seconds=15.0, source_fps=10.0)
    assert n_total == 3
    assert n_cropped == 2  # p1, n1; p2 is already short

    out_lines = out_path.read_text().strip().split("\n")
    assert len(out_lines) == 3
    out_records = [json.loads(line) for line in out_lines]
    assert out_records[0]["t_ref_frame"] == 75   # p1 remap
    assert out_records[1]["t_ref_frame"] is None  # n1
    assert out_records[2]["t_ref_frame"] == 42    # p2 unchanged
