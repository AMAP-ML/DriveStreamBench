"""Primary L3 F1 and gated accuracy share the reactive parsing contract."""

import pytest

from drivestreambench.metrics.l3_event import binary_f1_l3
from drivestreambench.metrics.l3_gated import (
    build_video_canonical_correct,
    is_l13_correct,
)


def test_invalid_l3_replies_count_as_errors_for_both_gold_classes():
    annotations = [
        {"question_id": "p", "question_level": "L3", "sample_type": "positive"},
        {"question_id": "n", "question_level": "L3", "sample_type": "negative"},
    ]
    predictions = [
        {"question_id": "p", "prediction": "Yesterday."},
        {"question_id": "n", "prediction": "NO REPLY"},
    ]
    assert binary_f1_l3(annotations, predictions) == {
        "f1": 0,
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "tn": 0,
        "n_total": 2,
    }


def test_legal_binary_decisions_preserve_the_confusion_matrix_and_f1():
    annotations, predictions = [], []
    for qid, gold, pred in [
        ("tp", "positive", "YES"),
        ("fn", "positive", "NO"),
        ("fp", "negative", "YES"),
        ("tn", "negative", "NO"),
    ]:
        annotations.append(
            {"question_id": qid, "question_level": "L3", "sample_type": gold}
        )
        predictions.append({"question_id": qid, "prediction": pred})
    assert binary_f1_l3(annotations, predictions) == {
        "f1": 0.5,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "n_total": 4,
    }


@pytest.mark.parametrize(
    "reply,gold,mcq,correct",
    [
        ("Yes.", "YES", False, True),
        ("No, it is not.", "NO", False, True),
        ("<think>...</think>YES<|im_end|>", "YES", False, True),
        ("Yesterday", "YES", False, False),
        ("NO REPLY", "NO", False, False),
        ("A)", "A", True, True),
        ("C. car", "C", True, True),
        ("The answer is C", "A", True, False),
        ("Because...", "B", True, False),
    ],
)
def test_gating_uses_only_a_normalized_answer(reply, gold, mcq, correct):
    assert is_l13_correct(reply, gold, mcq) is correct


def test_identified_gating_predictions_can_be_reordered_without_swapping_answers():
    records = [
        {
            "question_id": f"q{i}",
            "question_level": "L3",
            "answer": [{"content": gold}],
            "image_samples": [f"root/video{i}/CAM_F0/0.jpg"],
        }
        for i, gold in enumerate(["YES", "NO"])
    ]
    predictions = [
        {"question_id": "q1", "prediction": "NO"},
        {"question_id": "q0", "prediction": "YES"},
    ]
    assert build_video_canonical_correct(records, predictions) == {
        "video0": {"L3": True},
        "video1": {"L3": True},
    }


def test_auxiliary_per_class_recall_keeps_invalid_replies_in_its_denominator():
    from drivestreambench.metrics.l123_accuracy import score_l123

    annotations = [
        {"question_id": str(i), "question_level": "L3", "answer": "YES"}
        for i in range(2)
    ]
    predictions = [
        {"question_id": "0", "prediction": "YES"},
        {"question_id": "1", "prediction": "Yesterday."},
    ]
    binary = score_l123(annotations, predictions)["by_question_type"]["binary"]
    assert binary["n_scored"] == 2
    assert binary["per_class"]["YES"]["recall"] == 0.5
    assert binary["per_class"]["YES"]["f1"] == 0.666667
    assert "__INVALID__" not in binary["per_class"]
