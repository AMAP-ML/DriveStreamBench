"""End-to-end smoke test: toy annotations → run_l4 → score → matches expected."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def toy_l4_annotations() -> Path:
    return REPO / "examples" / "toy_annotations" / "L4_demo.jsonl"


@pytest.fixture
def toy_l4_predictions() -> Path:
    return REPO / "examples" / "toy_predictions" / "L4_demo_predictions.jsonl"


@pytest.fixture
def toy_l4_expected() -> Path:
    return REPO / "examples" / "expected_outputs" / "L4_demo_scores.json"


def test_score_toy_l4_matches_expected(toy_l4_annotations, toy_l4_predictions, toy_l4_expected):
    from drivestreambench.metrics.alertscore import score_l4_files

    actual = score_l4_files(str(toy_l4_annotations), str(toy_l4_predictions))
    expected = json.loads(toy_l4_expected.read_text())

    # Primary metric must match to 6 decimal places.
    assert actual["primary_metric"]["value"] == pytest.approx(
        expected["primary_metric"]["value"], abs=1e-6
    )
    # Per-subcategory cells must all match.
    for cat, blk in expected["per_subcategory"].items():
        assert cat in actual["per_subcategory"]
        for tau, cell in blk["by_tau"].items():
            a = actual["per_subcategory"][cat]["by_tau"][tau]
            assert a["alert_score"] == pytest.approx(cell["alert_score"], abs=1e-6)
            assert a["hit_rate"] == pytest.approx(cell["hit_rate"], abs=1e-6)


def test_score_toy_l123_matches_expected():
    from drivestreambench.metrics.l123_accuracy import score_l123_files

    ann = REPO / "examples" / "toy_annotations" / "L1L2L3_demo.jsonl"
    pred = REPO / "examples" / "toy_predictions" / "L1L2L3_demo_predictions.jsonl"
    expected = json.loads((REPO / "examples" / "expected_outputs" / "L1L2L3_demo_scores.json").read_text())

    actual = score_l123_files(str(ann), str(pred))
    assert actual["summary"]["accuracy"] == pytest.approx(
        expected["summary"]["accuracy"], abs=1e-6
    )
    for level, blk in expected["by_question_level"].items():
        assert actual["by_question_level"][level]["accuracy"] == pytest.approx(
            blk["accuracy"], abs=1e-6
        )


def test_run_l4_dummy_end_to_end(tmp_path):
    """Generate fresh predictions, score, must equal the bundled expected."""
    from drivestreambench.inference.run_l4 import run_l4
    from drivestreambench.models.dummy import AlertOnTurn
    from drivestreambench.metrics.alertscore import score_l4_files

    pred_path = tmp_path / "pred.jsonl"
    run_l4(
        annotations_path=str(REPO / "examples" / "toy_annotations" / "L4_demo.jsonl"),
        output_path=str(pred_path),
        model=AlertOnTurn(alert_turn=4),
        progress=False,
    )
    actual = score_l4_files(
        str(REPO / "examples" / "toy_annotations" / "L4_demo.jsonl"),
        str(pred_path),
    )
    expected = json.loads((REPO / "examples" / "expected_outputs" / "L4_demo_scores.json").read_text())
    assert actual["primary_metric"]["value"] == pytest.approx(
        expected["primary_metric"]["value"], abs=1e-6
    )
