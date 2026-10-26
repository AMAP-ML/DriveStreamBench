"""Reactive binary and MCQ questions use their corresponding system prompts."""

import copy
import json
from types import SimpleNamespace

from drivestreambench.inference.run_l123 import run_l123
from drivestreambench.models.api_vlm import APIVLM
from drivestreambench.protocol.prompts import (
    L1L2_MCQ_SYSTEM_PROMPT,
    L1L2L3_BINARY_SYSTEM_PROMPT,
    L4_SYSTEM_PROMPT,
)


def test_question_format_controls_system_prompt_and_resets_context(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    annotations = tmp_path / "questions.jsonl"
    annotations.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "question_id": "binary",
                    "question_level": "L1",
                    "video_name": "v",
                    "question_type": "vehicle",
                    "split": "test",
                    "question": "Is there a vehicle?",
                    "answer": "Yes",
                    "image_samples": [str(image)],
                },
                {
                    "question_id": "mcq",
                    "question_level": "L2",
                    "video_name": "v",
                    "question_type": "ego_lane",
                    "split": "test",
                    "question": "Choose A, B, C or D.",
                    "answer": "B",
                    "image_samples": [str(image)],
                },
            ]
        )
        + "\n"
    )
    calls = []
    replies = iter(["YES", "B"])

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))]
        )

    model = APIVLM(model_name="test")
    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    output = tmp_path / "predictions.jsonl"
    run_l123(
        annotations_path=annotations, output_path=output, model=model, progress=False
    )
    assert calls[0]["messages"][0]["content"] == L1L2L3_BINARY_SYSTEM_PROMPT
    assert calls[1]["messages"][0]["content"] == L1L2_MCQ_SYSTEM_PROMPT
    assert len(calls[0]["messages"]) == len(calls[1]["messages"]) == 2
    assert model.system_prompt == L4_SYSTEM_PROMPT
    assert [json.loads(s)["prediction"] for s in output.read_text().splitlines()] == [
        "YES",
        "B",
    ]
