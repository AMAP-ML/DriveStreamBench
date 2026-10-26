"""The left-aligned evaluation grid and terminal partial decision step."""

import json

import pytest

from drivestreambench.inference.run_l4 import run_l4
from drivestreambench.models.dummy import AlwaysSilent
from drivestreambench.protocol.frame_sampler import (
    decision_count,
    sample_session_grid,
    sample_turn_frames,
)
from drivestreambench.protocol.runner import SessionRunner


@pytest.mark.parametrize(
    "n_frames,expected_indices,expected_times",
    [
        (1, [[0]], [0.5]),
        (6, [[0, 5]], [1.0]),
        (11, [[0, 5], [10]], [1.0, 1.5]),
        (20, [[0, 5], [10, 15]], [1.0, 2.0]),
        (21, [[0, 5], [10, 15], [20]], [1.0, 2.0, 2.5]),
    ],
)
def test_uniform_groups_and_cumulative_processed_frame_times(
    n_frames, expected_indices, expected_times
):
    grid = sample_session_grid(decision_count(n_frames), n_total_frames=n_frames)
    assert [w.frame_indices for w in grid] == expected_indices
    assert [w.video_time for w in grid] == expected_times


def test_one_fps_uses_one_frame_per_second():
    grid = sample_session_grid(10, fps=1, frames_per_turn=1, n_total_frames=21)
    assert [w.frame_indices for w in grid] == [[0], [10], [20]]
    assert [w.video_time for w in grid] == [1, 2, 3]


def test_exhausted_frames_do_not_create_model_calls():
    session = SessionRunner(model=AlwaysSilent()).run(
        session_id="s",
        video_id="v",
        instruction="Watch",
        n_decisions=10,
        all_frame_paths=[str(i) for i in range(11)],
    )
    assert len(session.turns) == 2
    assert session.turns[-1].frame_indices == [10]
    assert session.turns[-1].video_time == 1.5


def test_cli_default_keeps_last_partial_group_instead_of_rounding_duration(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "question_id": "s",
                "question_level": "L4",
                "video_name": "v",
                "question_type": "lead_vehicle_start",
                "sample_type": "negative",
                "question": "Watch",
                "image_samples": [str(i) for i in range(11)],
                "alert_mode": "event_triggered",
                "split": "test",
            }
        )
        + "\n"
    )
    output = tmp_path / "pred.jsonl"
    run_l4(
        annotations_path=annotations,
        output_path=output,
        model=AlwaysSilent(),
        progress=False,
    )
    turns = json.loads(output.read_text())["turns"]
    assert [t["video_time"] for t in turns] == [1, 1.5]


@pytest.mark.parametrize(
    "parameters",
    [
        {"fps": 0},
        {"source_fps": 0},
        {"frames_per_turn": 0},
        {"decision_cadence_hz": 0},
        {"fps": 1, "frames_per_turn": 2},
    ],
)
def test_invalid_grid_parameters_fail_before_inference(parameters):
    with pytest.raises(ValueError):
        sample_turn_frames(0, **parameters)
