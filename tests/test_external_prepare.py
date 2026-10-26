"""Source-grid, reference-coordinate and native-format preparation contracts."""
import copy
import json

import pytest

from external_baselines.native import frame_paths, validate_inputs
from external_baselines.prepare import prepare_files, prepare_records
from drivestreambench.metrics.alertscore import build_outcomes
from drivestreambench.protocol.prompts import L4_SYSTEM_PROMPT


def annotation(n=21, anchor=9, mode="target_timing"):
    return {"question_id": "L4_reference", "question_level": "L4",
            "question_type": "lead_vehicle_start", "sample_type": "positive",
            "alert_mode": mode, "question": "Please watch this exact instruction!",
            "t_ref_frame": anchor, "image_samples": [f"video/{i:06d}.jpg" for i in range(n)]}


@pytest.mark.parametrize("fps,expected_indices,group_sizes", [(1, [0, 10, 20], [1, 1, 1]), (2, [0, 5, 10, 15, 20], [2, 2, 1])])
def test_native_frames_initial_instruction_and_terminal_partial_group(tmp_path, fps, expected_indices, group_sizes):
    row = annotation()
    original = copy.deepcopy(row)
    inputs, golds, scoring = prepare_records([row], dataroot=tmp_path, fps=fps, check_images=False)
    sample = inputs[0]
    assert row == original
    assert sample["question_id"] == row["question_id"]
    assert sample["conversation"][0] == {"role": "system", "content": L4_SYSTEM_PROMPT}
    assert frame_paths(sample) == [str(tmp_path / f"video/{i:06d}.jpg") for i in expected_indices]
    users = sample["conversation"][1:]
    assert [sum(item["type"] == "image" for item in turn["content"]) for turn in users] == group_sizes
    texts = [item["text"] for turn in users for item in turn["content"] if item["type"] == "text"]
    assert texts == [row["question"]]
    assert users[0]["content"][-1] == {"type": "text", "text": row["question"]}
    assert golds[0]["answer"][0]["alert_mode"] == scoring[0]["alert_mode"] == "target_timing"
    assert "t_ref_frame" not in scoring[0]


@pytest.mark.parametrize("fps,n,anchor,expected", [(1, 21, 1, 1.0), (2, 21, 1, 0.5), (1, 20, 19, 1.0), (2, 20, 19, 1.5), (2, 1, 0, 0.0)])
def test_reference_uses_next_grid_frame_and_clamps_to_final_sample(tmp_path, fps, n, anchor, expected):
    inputs, golds, scoring = prepare_records([annotation(n, anchor)], dataroot=tmp_path, fps=fps, check_images=False)
    assert golds[0]["answer"][0]["t_ref"] == scoring[0]["t_ref"] == expected
    predictions = [{"question_id": "L4_reference", "turns": [{"video_time": 1.0, "reply": "ALERT"}]}]
    assert build_outcomes(scoring, predictions)[0].t_ref == expected
    # The reference label never selects or repositions observed frames.
    other = annotation(n, 0)
    assert inputs == prepare_records([other], dataroot=tmp_path, fps=fps, check_images=False)[0]


def test_negative_and_modes_are_preserved(tmp_path):
    row = annotation(mode="event_triggered")
    row.update(sample_type="negative", t_ref_frame=None)
    _, golds, scoring = prepare_records([row], dataroot=tmp_path, fps=1, check_images=False)
    assert golds[0]["answer"] == [{"content": "NO REPLY", "t_ref": None, "alert_mode": "event_triggered"}]
    assert scoring[0]["sample_type"] == "negative" and scoring[0]["t_ref"] is None


def test_file_preparation_is_native_runner_compatible_and_preserves_source(tmp_path):
    row = annotation(11, 9)
    for index in (0, 5, 10):
        image = tmp_path / row["image_samples"][index]
        image.parent.mkdir(exist_ok=True)
        image.touch()
    source = tmp_path / "annotations.jsonl"
    source.write_text(json.dumps(row) + "\n")
    before = source.read_bytes()
    output = tmp_path / "prepared"
    assert prepare_files(annotations=source, dataroot=tmp_path, fps=2, output_dir=output) == 1
    assert len(validate_inputs(output / "input.json", output / "gold.json")) == 1
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_files(annotations=source, dataroot=tmp_path, fps=2, output_dir=output)


def test_invalid_records_fail_without_writing_output(tmp_path):
    row = annotation()
    with pytest.raises(ValueError, match="duplicate"):
        prepare_records([row, row], dataroot=tmp_path, fps=1, check_images=False)
    for change in ({"question_level": "L3"}, {"t_ref_frame": 99}, {"image_samples": ["../outside.jpg"]}, {"image_samples": ["/outside.jpg"]}):
        with pytest.raises(ValueError):
            prepare_records([{**row, **change}], dataroot=tmp_path, fps=1, check_images=False)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(FileNotFoundError):
        prepare_files(annotations=source, dataroot=tmp_path, fps=1, output_dir=tmp_path / "not_written")
    assert not (tmp_path / "not_written").exists()
