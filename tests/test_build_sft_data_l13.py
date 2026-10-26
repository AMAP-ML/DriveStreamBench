"""Unit tests for :mod:`drivestreambench.training.build_sft_data_l13`.

Covers:

* L1 binary record → single-turn ``messages`` with binary system prompt
* L2 MCQ record    → single-turn ``messages`` with MCQ system prompt
* L3 binary record → single-turn ``messages`` with binary system prompt
* L4 records are skipped (returns ``None``)
* Records with no answer are skipped
* Frame sub-sampling caps at ``--max-frames``
* ``--dataroot`` rewrites image paths
* Multi-file build merges + counts correctly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from drivestreambench.data.schemas import Sample
from drivestreambench.training.build_sft_data_l13 import (
    _uniform_subsample,
    build,
    to_l13_record,
)


def _ann_l1(qid: str, answer: str = "Yes", n_imgs: int = 3) -> Dict[str, Any]:
    return {
        "question_id": qid,
        "question_level": "L1",
        "video_name": "video_a",
        "question_type": "vehicle",
        "split": "test",
        "sample_type": None,
        "alert_mode": None,
        "source_type": "nuplan",
        "image_samples": [f"video_a/CAM_F0/{i:06d}.jpg" for i in range(n_imgs)],
        "t_ref_frame": None,
        "question": "Is there a vehicle ahead?",
        "answer": answer,
        "trigger_candidate_count": None,
    }


def _ann_l2(qid: str, answer: str = "B") -> Dict[str, Any]:
    rec = _ann_l1(qid, answer=answer)
    rec.update({"question_level": "L2", "question_type": "ego_lane"})
    return rec


def _ann_l3(qid: str, answer: str = "Yes", n_imgs: int = 50) -> Dict[str, Any]:
    rec = _ann_l1(qid, answer=answer, n_imgs=n_imgs)
    rec.update(
        {
            "question_level": "L3",
            "question_type": "lead_vehicle_brake",
            "sample_type": "positive",
        }
    )
    return rec


def _ann_l4(qid: str = "L4_x") -> Dict[str, Any]:
    return {
        "question_id": qid,
        "question_level": "L4",
        "video_name": "video_a",
        "question_type": "lead_vehicle_brake",
        "split": "test",
        "sample_type": "positive",
        "alert_mode": "event_triggered",
        "source_type": "nuplan",
        "image_samples": ["video_a/CAM_F0/000000.jpg"],
        "t_ref_frame": 0,
        "question": "Notify…",
        "answer": None,
        "trigger_candidate_count": 1,
    }


# ---- to_l13_record per-level shape -----------------------------------------


class TestPerLevelShape:
    def test_l1_uses_binary_system(self):
        rec = to_l13_record(Sample.from_dict(_ann_l1("L1_a")))
        assert rec is not None
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert (
            "YES" in rec["messages"][0]["content"]
            and "NO" in rec["messages"][0]["content"]
        )
        assert rec["messages"][2]["content"] == "YES"
        assert rec["level"] == "L1"

    def test_l2_uses_mcq_system(self):
        rec = to_l13_record(Sample.from_dict(_ann_l2("L2_a", answer="C")))
        assert rec is not None
        sys_text = rec["messages"][0]["content"]
        assert "option letter" in sys_text or "A, B, C, or D" in sys_text
        assert rec["messages"][2]["content"] == "C"
        assert rec["level"] == "L2"

    def test_l3_uses_binary_system(self):
        rec = to_l13_record(Sample.from_dict(_ann_l3("L3_a", answer="No")))
        assert rec is not None
        assert rec["messages"][2]["content"] == "NO"
        assert rec["level"] == "L3"

    def test_l4_returns_none(self):
        assert to_l13_record(Sample.from_dict(_ann_l4("L4_x"))) is None

    def test_empty_answer_returns_none(self):
        assert to_l13_record(Sample.from_dict(_ann_l1("x", answer=""))) is None


# ---- frame subsampling -----------------------------------------------------


class TestFrameSubsampling:
    def test_subsample_evenly_spaced(self):
        # 10 frames → keep 4 → indices [0, 3, 6, 9]
        idxs = _uniform_subsample(10, 4)
        assert idxs == [0, 3, 6, 9]

    def test_subsample_no_op_when_smaller(self):
        assert _uniform_subsample(2, 8) == [0, 1]

    def test_max_frames_caps_image_count(self):
        rec = to_l13_record(Sample.from_dict(_ann_l3("x", n_imgs=50)), max_frames=5)
        assert rec is not None
        assert len(rec["images"]) == 5
        # The standard ms-swift text uses one placeholder per image path.
        user_content = rec["messages"][1]["content"]
        assert isinstance(user_content, str)
        assert user_content.count("<image>") == len(rec["images"]) == 5


# ---- dataroot prefix -------------------------------------------------------


def test_dataroot_rewrites_image_paths():
    rec = to_l13_record(
        Sample.from_dict(_ann_l1("x", n_imgs=2)),
        dataroot="/abs/dataroot",
    )
    assert rec is not None
    assert all(p.startswith("/abs/dataroot/") for p in rec["images"])
    assert rec["images"][0].endswith("000000.jpg")


def test_no_dataroot_keeps_relative_paths():
    rec = to_l13_record(Sample.from_dict(_ann_l1("x", n_imgs=2)))
    assert rec is not None
    assert all(not p.startswith("/") for p in rec["images"])


# ---- multi-file build ------------------------------------------------------


def test_build_merges_three_levels(tmp_path: Path):
    p_l1 = tmp_path / "l1_train.jsonl"
    p_l2 = tmp_path / "l2_train.jsonl"
    p_l3 = tmp_path / "l3_train.jsonl"
    p_l1.write_text(json.dumps(_ann_l1("L1_a")) + "\n", encoding="utf-8")
    p_l2.write_text(json.dumps(_ann_l2("L2_a")) + "\n", encoding="utf-8")
    p_l3.write_text(json.dumps(_ann_l3("L3_a")) + "\n", encoding="utf-8")
    out = tmp_path / "l13_train.jsonl"
    n = build(annotations_paths=[p_l1, p_l2, p_l3], output_path=out, progress=False)
    assert n == 3
    levels = [json.loads(line)["level"] for line in out.read_text().splitlines()]
    assert sorted(levels) == ["L1", "L2", "L3"]


def test_build_skips_l4_records(tmp_path: Path):
    """A mixed input with both L4 and L3 records emits only the L3 record."""
    p_mixed = tmp_path / "mixed.jsonl"
    p_mixed.write_text(
        json.dumps(_ann_l4()) + "\n" + json.dumps(_ann_l3("L3_b")) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "mixed_l13.jsonl"
    n = build(annotations_paths=[p_mixed], output_path=out, progress=False)
    assert n == 1
    assert json.loads(out.read_text().splitlines()[0])["level"] == "L3"
