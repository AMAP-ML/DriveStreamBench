"""The public scorer uses FPS-specific references and preserves explicit seconds."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from drivestreambench.metrics.alertscore import build_outcomes, reference_time, score_l4
from drivestreambench.metrics.l3_gated import score_l3_gated
from drivestreambench.metrics.tau_curves import score_l4_tau_grid


def annotation(n=21, anchor=1):
    return {
        "question_id": "2021.01.01.10.00.00_veh-01_0001_0100_lead_vehicle_brake_0001",
        "question_type": "lead_vehicle_brake",
        "sample_type": "positive",
        "alert_mode": "event_triggered",
        "t_ref_frame": anchor,
        "image_samples": [f"video/{i}.jpg" for i in range(n)],
    }


def prediction(row):
    return {
        "question_id": row["question_id"],
        "turns": [{"video_time": 0.5, "reply": "ALERT"}],
    }


@pytest.mark.parametrize(
    "fps,n,anchor,expected",
    [
        (2, 21, 1, 0.5),
        (1, 21, 1, 1.0),
        (2, 21, 10, 1.0),
        (2, 20, 19, 1.5),
        (1, 20, 19, 1.0),
        (2, 1, 0, 0.0),
    ],
)
def test_reference_matches_the_original_ceil_and_terminal_clamp(
    fps, n, anchor, expected
):
    assert reference_time(annotation(n, anchor), fps=fps) == expected


def test_explicit_seconds_are_never_requantized_even_with_a_source_anchor():
    row = annotation()
    row["t_ref"] = 0.123
    assert reference_time(row, fps=1) == reference_time(row, fps=2) == 0.123
    row.pop("image_samples")
    row.pop("t_ref_frame")
    assert reference_time(row) == 0.123


@pytest.mark.parametrize(
    "row",
    [
        {"t_ref_frame": 1},
        {"t_ref_frame": 2, "image_samples": ["0", "1"]},
        {"t_ref_frame": 0.2, "image_samples": ["0", "1"]},
    ],
)
def test_source_reference_requires_the_source_clip_for_correct_clamping(row):
    with pytest.raises(ValueError):
        reference_time(row)


def test_default_metric_and_gating_share_the_same_reference_grid():
    row = annotation()
    pred = prediction(row)
    assert build_outcomes([row], [pred])[0].t_ref == 0.5
    assert score_l4([row], [pred])["primary_metric"]["value"] == 1
    assert score_l4_tau_grid([row], [pred], fps=1)["primary_metric"]["value"] == 0
    video = "2021.01.01.10.00.00_veh-01_0001_0100"
    assert (
        score_l3_gated(
            build_outcomes([row], [pred]), video_correct={video: {"L3": True}}
        ).macro_grounded_hit_at_tau
        == 1
    )
    assert (
        score_l3_gated(
            build_outcomes([row], [pred], fps=1), video_correct={video: {"L3": True}}
        ).macro_grounded_hit_at_tau
        == 0
    )


def test_cli_default_and_explicit_fps_use_the_same_public_scoring_path(tmp_path):
    row = annotation()
    annotations, predictions = tmp_path / "ann.jsonl", tmp_path / "pred.jsonl"
    annotations.write_text(json.dumps(row) + "\n")
    predictions.write_text(json.dumps(prediction(row)) + "\n")
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
    for arguments, expected in [([], 1), (["--fps", "1"], 0)]:
        output = tmp_path / f"score{expected}.json"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--annotations",
                str(annotations),
                "--predictions",
                str(predictions),
                "--output",
                str(output),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(output.read_text())
        assert result["primary_metric"]["value"] == expected
        assert result["fps"] == (1 if arguments else 2)


def test_scalar_reference_without_mode_uses_the_canonical_category_dispatch():
    row = {
        "question_id": "p",
        "subcategory": "adjacent_cut_in",
        "is_positive": True,
        "t_ref": 1.0,
    }
    assert build_outcomes([row], [prediction(row)])[0].alert_mode == "target_timing"
