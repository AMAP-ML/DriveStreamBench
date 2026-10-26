"""Check the reference recipe and the exact data consumed by ms-swift."""

import random

import pytest

from drivestreambench.data.schemas import Sample
from drivestreambench.protocol.prompts import (
    L1L2_MCQ_SYSTEM_PROMPT,
    L1L2L3_BINARY_SYSTEM_PROMPT,
)
from drivestreambench.training import train_sft
from drivestreambench.training.build_sft_data import to_ms_swift_record
from drivestreambench.training.build_sft_data_l13 import to_l13_record


def sample(level="L4", answer=None):
    return Sample(
        question_id="sample",
        question_level=level,
        video_name="video",
        question_type="lead_vehicle_start",
        split="train",
        sample_type="positive",
        alert_mode="event_triggered",
        image_samples=[f"video/CAM_F0/{i}.jpg" for i in range(31)],
        t_ref_frame=10,
        question="Watch the vehicle",
        answer=answer,
    )


def test_l4_loss_masks_are_message_fields_consumed_by_swift():
    record = to_ms_swift_record(sample(), loss_mask_rate=1.0, rng=random.Random(0))
    assistants = [m for m in record["messages"] if m["role"] == "assistant"]
    assert sum(m["content"] == "ALERT" for m in assistants) == 1
    assert all(m["loss"] == (m["content"] == "ALERT") for m in assistants)
    user_messages = [m["content"] for m in record["messages"] if m["role"] == "user"]
    assert all(isinstance(text, str) for text in user_messages)
    assert sum(text.count("<image>") for text in user_messages) == len(record["images"])


def test_l13_default_preserves_frames_and_normalizes_answer():
    record = to_l13_record(sample("L3", "Yes"))
    assert len(record["images"]) == 31
    assert record["messages"][1]["content"].count("<image>") == 31
    assert record["messages"][-1] == {
        "role": "assistant",
        "content": "YES",
        "loss": True,
    }


@pytest.mark.parametrize(
    "level,answer,expected",
    [("L1", "A", L1L2_MCQ_SYSTEM_PROMPT), ("L2", "No", L1L2L3_BINARY_SYSTEM_PROMPT)],
)
def test_l13_prompt_uses_question_format_instead_of_level(level, answer, expected):
    assert to_l13_record(sample(level, answer))["messages"][0]["content"] == expected


def test_reference_hyperparameters_reach_actual_backend_command(tmp_path):
    args = train_sft.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "dry-run",
        ]
    )
    cmd = train_sft.build_command(args)
    for name, value in {
        "--max_length": "40960",
        "--adam_beta1": "0.9",
        "--adam_beta2": "0.95",
        "--weight_decay": "0.1",
        "--torch_dtype": "bfloat16",
    }.items():
        assert cmd[cmd.index(name) + 1] == value
    assert cmd[:2] == ["swift", "sft"]
    assert train_sft.build_environment(args)["IMAGE_MAX_TOKEN_NUM"] == "256"


def test_image_budget_is_forwarded_to_subprocess(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_sft.shutil, "which", lambda _: "/bin/swift")
    monkeypatch.setattr(
        train_sft.subprocess,
        "call",
        lambda command, **kwargs: calls.append((command, kwargs)) or 0,
    )
    with pytest.raises(SystemExit) as exc:
        train_sft.main(
            [
                "--train-jsonl",
                "train.jsonl",
                "--output-dir",
                str(tmp_path),
                "--image-max-tokens",
                "128",
            ]
        )
    assert exc.value.code == 0
    assert calls[0][1]["env"]["IMAGE_MAX_TOKEN_NUM"] == "128"


def test_dry_run_shows_the_real_command_without_launching(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        train_sft.subprocess,
        "call",
        lambda *args, **kwargs: pytest.fail("dry-run launched training"),
    )
    train_sft.main(
        [
            "--train-jsonl",
            "train.jsonl",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert "swift sft" in output and "--max_length 40960" in output
    assert "IMAGE_MAX_TOKEN_NUM=256" in output


def test_sft_alert_follows_the_observed_anchor_group():
    from drivestreambench.training.build_sft_data import session_to_turns

    turns = session_to_turns(sample())
    assert turns[0]["frame_idxs"] == [0, 5]
    assert turns[0]["content"] == "NO REPLY"
    assert turns[1]["frame_idxs"] == [10, 15]
    assert turns[1]["content"] == "ALERT"
    assert turns[-1]["frame_idxs"] == [30]


def test_training_grid_keeps_an_off_grid_reference_frame():
    from drivestreambench.training.build_sft_data import session_to_turns

    record = sample()
    record.t_ref_frame = 12
    turns = session_to_turns(record)
    all_indices = [idx for turn in turns for idx in turn["frame_idxs"]]
    assert all_indices == [0, 2, 7, 12, 17, 22, 27]
    assert [t["frame_idxs"] for t in turns if t["content"] == "ALERT"] == [[7, 12]]
