"""Tests for :mod:`drivestreambench.metrics.l123_accuracy`."""
from __future__ import annotations

import pytest

from drivestreambench.metrics.l123_accuracy import normalize_reply, score_l123


def test_normalize_reply_binary_yes_no():
    assert normalize_reply("Yes", "binary") == ("YES", "ok")
    assert normalize_reply("no", "binary") == ("NO", "ok")
    assert normalize_reply("YES.", "binary") == ("YES", "ok")
    assert normalize_reply(" Yes ", "binary") == ("YES", "ok")


def test_normalize_reply_binary_yes_with_explanation():
    """Some models prefix the answer with the option then explain — the
    parser must lock onto the YES/NO prefix and ignore the rest."""
    assert normalize_reply("Yes, the lead vehicle is moving.", "binary") == ("YES", "ok")
    assert normalize_reply("No — there is no pedestrian.", "binary") == ("NO", "ok")


def test_normalize_reply_mcq_letter():
    assert normalize_reply("A", "mcq") == ("A", "ok")
    assert normalize_reply("B.", "mcq") == ("B", "ok")
    assert normalize_reply("c) the leftmost vehicle", "mcq") == ("C", "ok")


def test_normalize_reply_no_reply_status():
    assert normalize_reply("NO REPLY", "binary") == ("NO REPLY", "no_reply")


def test_normalize_reply_invalid():
    assert normalize_reply("dunno", "binary")[1] == "invalid"
    assert normalize_reply("E", "mcq")[1] == "invalid"


def test_normalize_reply_missing():
    assert normalize_reply(None, "binary") == (None, "missing")
    assert normalize_reply("", "mcq") == (None, "missing")


def _ann_l1(qid, ans):
    return {
        "question_id": qid, "question_level": "L1", "video_name": "v",
        "question_type": "vehicle", "split": "test", "sample_type": None,
        "alert_mode": None, "source_type": "nuplan", "image_samples": [],
        "t_ref_frame": None, "question": "q?", "answer": ans,
        "trigger_candidate_count": None, "metadata": {"original_question_id": qid},
    }


def test_score_l123_perfect_reactive_run():
    annotations = [_ann_l1("a", "Yes"), _ann_l1("b", "No")]
    predictions = [{"question_id": "a", "prediction": "Yes"},
                   {"question_id": "b", "prediction": "No"}]
    r = score_l123(annotations, predictions)
    assert r["summary"]["accuracy"] == 1.0
    assert r["by_question_level"]["L1"]["accuracy"] == 1.0


def test_score_l123_all_wrong():
    annotations = [_ann_l1("a", "Yes"), _ann_l1("b", "No")]
    predictions = [{"question_id": "a", "prediction": "No"},
                   {"question_id": "b", "prediction": "Yes"}]
    r = score_l123(annotations, predictions)
    assert r["summary"]["accuracy"] == 0.0


def test_score_l123_uses_release_field_names():
    """`question_level` (not `level`) and `prediction` (not `reply`)."""
    annotations = [_ann_l1("a", "Yes")]
    predictions = [{"question_id": "a", "prediction": "Yes"}]
    r = score_l123(annotations, predictions)
    assert r["summary"]["accuracy"] == 1.0


def test_score_l123_macro_f1_per_class():
    """All gold YES, model always says YES → YES has perfect P/R/F1; NO has zero."""
    annotations = [_ann_l1(f"q{i}", "Yes") for i in range(5)]
    predictions = [{"question_id": f"q{i}", "prediction": "Yes"} for i in range(5)]
    r = score_l123(annotations, predictions)
    binary = r["by_question_type"]["binary"]
    assert binary["per_class"]["YES"]["f1"] == 1.0
    # NO never appears in either gold or predictions → F1 undefined → 0
    assert binary["per_class"].get("NO", {}).get("f1", 0.0) == 0.0
