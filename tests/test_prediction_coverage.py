"""Incomplete or ambiguous prediction records cannot receive benchmark credit."""

import pytest

from drivestreambench.metrics.alertscore import score_l4


def records():
    annotations = [
        {
            "question_id": "positive",
            "question_type": "lead_vehicle_start",
            "sample_type": "positive",
            "t_ref_frame": 10,
            "image_samples": [str(i) for i in range(21)],
            "alert_mode": "event_triggered",
        },
        {
            "question_id": "negative",
            "question_type": "lead_vehicle_start",
            "sample_type": "negative",
            "t_ref_frame": None,
            "alert_mode": "event_triggered",
        },
    ]
    predictions = [
        {"question_id": "positive", "turns": [{"video_time": 1.0, "reply": "ALERT"}]},
        {
            "question_id": "negative",
            "turns": [{"video_time": 1.0, "reply": "NO REPLY"}],
        },
    ]
    return annotations, predictions


def test_missing_negative_prediction_is_rejected_instead_of_scored_silent():
    annotations, predictions = records()
    with pytest.raises(ValueError, match="1 missing"):
        score_l4(annotations, predictions[:1])


def test_duplicate_predictions_are_rejected():
    annotations, predictions = records()
    with pytest.raises(ValueError, match="Duplicate prediction"):
        score_l4(annotations, predictions + predictions[:1])


def test_duplicate_annotation_ids_are_rejected():
    annotations, predictions = records()
    with pytest.raises(ValueError, match="Duplicate annotation"):
        score_l4(annotations + annotations[:1], predictions)


def test_unexpected_prediction_ids_are_rejected():
    annotations, predictions = records()
    with pytest.raises(ValueError, match="1 unexpected"):
        score_l4(annotations, predictions + [{"question_id": "other", "turns": []}])


def test_complete_predictions_keep_existing_metric_definition():
    annotations, predictions = records()
    assert score_l4(annotations, predictions)["primary_metric"]["value"] == 1.0
