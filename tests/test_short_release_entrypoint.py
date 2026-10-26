"""Regression coverage for the public short-subset shell entry point."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from drivestreambench.data.validate import validate


ROOT = Path(__file__).resolve().parents[1]


def public_record(qid, sample_type):
    """Construct a record with the 13 published fields."""
    return {
        "question_id": qid,
        "question_level": "L4",
        "video_name": "clip",
        "question_type": "lead_vehicle_start",
        "split": "test",
        "sample_type": sample_type,
        "alert_mode": "target_timing",
        "source_type": "nuplan",
        "image_samples": [f"clip/CAM_F0/{i:04d}.jpg" for i in range(300)],
        "t_ref_frame": 200 if sample_type == "positive" else None,
        "question": "Alert me when the front vehicle starts moving.",
        "answer": None,
        "trigger_candidate_count": 1 if sample_type == "positive" else 0,
    }


@pytest.mark.parametrize("seconds,frames", [(None, 150), ("5", 50)])
def test_shell_duration_and_public_crop_schema(tmp_path, seconds, frames):
    records = [public_record("positive", "positive"), public_record("negative", "negative")]
    source = tmp_path / "source.jsonl"
    original = "".join(json.dumps(r) + "\n" for r in records)
    source.write_text(original)
    output = tmp_path / "short.jsonl"
    env = os.environ.copy()
    env.update(INPUT=str(source), OUTPUT=str(output), SOURCE_FPS="10", SECONDS="0",
               PYTHONPATH=str(ROOT), PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""))
    env.pop("DSB_SHORT_SECONDS", None)
    if seconds is not None:
        env["DSB_SHORT_SECONDS"] = seconds
    subprocess.run(["bash", str(ROOT / "scripts/build_l4_short.sh")], cwd=ROOT,
                   env=env, check=True, capture_output=True, text=True)
    cropped = [json.loads(line) for line in output.read_text().splitlines()]
    assert source.read_text() == original
    assert len(cropped) == 2
    assert all(len(r["image_samples"]) == frames for r in cropped)
    positive, negative = cropped
    assert positive["image_samples"][positive["t_ref_frame"]] == records[0]["image_samples"][200]
    assert negative["image_samples"] == records[1]["image_samples"][:frames]
    assert [r["question_id"] for r in cropped] == [r["question_id"] for r in records]
    assert "original_question_id" not in positive["metadata"]
    assert validate(cropped) == []
