"""Launch native model inference and retain its raw, timestamped responses."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BASELINES = ("dispider", "streamforest", "simplestream", "videollm_online", "mmduet2", "m4_omnimmi")
M4_TRIGGER = re.compile(r"REQUIREMENT MEET AT\s*([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


def metadata(name: str) -> dict[str, Any]:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    return json.loads((ROOT / f"{name}.yaml").read_text(encoding="utf-8"))


def _unique_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must contain a nonempty JSON array")
    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in by_id:
            raise ValueError(f"{label}: missing or duplicate question_id: {qid!r}")
        by_id[qid] = row
    return by_id


def validate_inputs(input_path: Path, gold_path: Path) -> list[dict[str, Any]]:
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    golds = json.loads(gold_path.read_text(encoding="utf-8"))
    inputs_by_id = _unique_rows(inputs, "input")
    golds_by_id = _unique_rows(golds, "gold")
    if inputs_by_id.keys() != golds_by_id.keys():
        raise ValueError("Input and gold must have exactly the same question IDs")
    for row in inputs:
        paths = frame_paths(row)
        if not paths:
            raise ValueError(f"No frames for {row['question_id']}")
        for path in paths:
            if not Path(path).is_absolute():
                raise ValueError(f"Native frame inputs require absolute image paths: {path}")
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return inputs


def frame_paths(sample: dict[str, Any]) -> list[str]:
    return [
        item["image"]
        for turn in sample.get("conversation", [])
        if turn.get("role") == "user" and isinstance(turn.get("content"), list)
        for item in turn["content"]
        if isinstance(item, dict) and item.get("type") == "image"
    ]


def verify_install(name: str, install_dir: Path) -> None:
    meta = metadata(name)
    resolved = subprocess.check_output(
        ["git", "-C", str(install_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved != meta["upstream"]["commit"]:
        raise ValueError(f"{name}: expected upstream revision {meta['upstream']['commit']}, got {resolved}")
    patch_path = ROOT / "patches" / f"{name}.patch"
    if patch_path.exists():
        subprocess.run(
            ["git", "-C", str(install_dir), "apply", "--reverse", "--check", str(patch_path)],
            check=True, capture_output=True, text=True,
        )
    for path in meta["required_files"]:
        if not (install_dir / path).is_file():
            raise FileNotFoundError(install_dir / path)
    if name == "dispider":
        from external_baselines.compatibility import verify_dispider_source
        verify_dispider_source(install_dir)


def build_command(
    name: str, *, install_dir: Path, checkpoint: str, input_path: Path,
    gold_path: Path, native_output: Path, fps: int = 2, gpus: str = "0",
    llm: str | None = None, vision: str | None = None, compressor: str | None = None,
    python: str = sys.executable,
) -> tuple[list[str], dict[str, str], Path]:
    """Construct each model's actual CLI; no inference runs in this function."""
    if fps not in (1, 2):
        raise ValueError("Supported input rates are 1 and 2 FPS")
    ids = gpus.split(",")
    if not ids or any(not item.isdigit() for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("--gpus must be distinct comma-separated GPU indices")
    meta = metadata(name)
    command = [python, str(ROOT / "native_inference" / f"{name}.py")]
    interval = str(1.0 / fps)
    if name == "videollm_online":
        if not llm or not vision:
            raise ValueError("VideoLLM-online requires --llm and --vision as well as --checkpoint")
        command += ["--resume_from_checkpoint", checkpoint, "--llm_pretrained", llm,
                    "--vision_pretrained", vision, "--input_file", str(input_path),
                    "--gold_file", str(gold_path), "--output_file", str(native_output),
                    "--gpu_id", "0", "--frame_interval", interval, "--frame_fps", str(fps)]
    else:
        weight_flag = "--llm_pretrained" if name == "mmduet2" else "--model_path"
        command += [weight_flag, checkpoint, "--test_fname", str(input_path),
                    "--gold_fname", str(gold_path), "--output_fname", str(native_output)]
        gpu_flag = "--gpu_ids" if name in ("streamforest", "mmduet2") else "--gpu_id"
        logical_gpus = ",".join(str(index) for index in range(len(ids))) if name == "mmduet2" else "0"
        command += [gpu_flag, logical_gpus]
        command += ["--fps", str(fps)] if name == "streamforest" else ["--frame_interval", interval]
        if name == "mmduet2":
            command += ["--output_dir", str(native_output.parent / "model_output")]
    command += ["--alert_match_mode", "strict" if name == "m4_omnimmi" else "lenient"]
    command += [str(arg) for arg in meta.get("default_args", [])]
    env = os.environ.copy()
    paths = [str(install_dir), str(ROOT / "shared"), str(install_dir / "proactive_eval"),
             str(install_dir / "train"), str(install_dir / "intersuit")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.pop("LOCAL_RANK", None)
    env.pop("WORLD_SIZE", None)
    selected_vision = vision or meta.get("vision_checkpoint")
    if selected_vision:
        env["DSB_VISION_CHECKPOINT"] = selected_vision
    if name == "dispider":
        env["DSB_DISPIDER_COMPRESSOR"] = compressor or str(Path(checkpoint) / "stream_compressor")
    if name == "m4_omnimmi":
        env["M4_CHECKPOINT_ROOT"] = checkpoint
        if selected_vision:
            env["M4_VISION_CHECKPOINT"] = selected_vision
    cwd = install_dir / meta.get("working_directory", ".")
    return command, env, cwd


def convert_raw_outputs(
    name: str, inputs: list[dict[str, Any]], raw_rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert complete native logs without changing free-form action text.

    M4 uses its native intersuit trigger and its reported target time. The
    observation time remains in each turn's extra field.
    """
    expected = _unique_rows(inputs, "input")
    logs = _unique_rows(raw_rows, "raw outputs")
    native = _unique_rows(native_rows, "native outputs")
    if expected.keys() != logs.keys() or expected.keys() != native.keys():
        raise ValueError("Incomplete native output coverage or unexpected question IDs")
    converted: list[dict[str, Any]] = []
    for qid in expected:
        row = logs[qid]
        if row.get("error") or row.get("errors"):
            raise RuntimeError(f"Native inference failed for {qid}: {row.get('error') or row.get('errors')}")
        if row.get("frame_count") != len(frame_paths(expected[qid])):
            raise ValueError(f"Frame-count mismatch for {qid}")
        events = row.get("raw_model_outputs")
        if not isinstance(events, list):
            raise ValueError(f"Missing raw_model_outputs for {qid}")
        turns: list[dict[str, Any]] = []
        for event in events:
            if event.get("role") not in (None, "assistant"):
                continue
            timestamp = event.get("time")
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"Invalid native timestamp for {qid}: {timestamp!r}")
            raw = event.get("raw_output", event.get("content"))
            if raw is not None and not isinstance(raw, str):
                raise ValueError(f"Invalid raw response for {qid}: {type(raw).__name__}")
            extra: dict[str, Any] = {"native_time": timestamp, "raw_output": raw}
            if name == "m4_omnimmi":
                trigger = M4_TRIGGER.search(raw or "")
                if trigger:
                    reply, timestamp = "ALERT", float(trigger.group(1))
                    extra["native_trigger"] = "intersuit"
                else:
                    reply = "NO REPLY"
            else:
                reply = raw if raw is not None else ""
            turns.append({"turn_idx": len(turns), "video_time": timestamp, "reply": reply, "extra": extra})
        if not turns:
            raise ValueError(f"No native assistant decisions for {qid}")
        converted.append({"question_id": qid, "model_name": name, "turns": turns,
                          "extra": {"inference_adapter": "native_intersuit" if name == "m4_omnimmi" else "native_batch",
                                    "parser": "strict_literal" if name == "m4_omnimmi" else "action_level"}})
    return converted


def run_native(
    name: str, *, install_dir: Path, checkpoint: str, input_path: Path,
    gold_path: Path, output_path: Path, fps: int | None = None, gpus: str = "0",
    llm: str | None = None, vision: str | None = None, compressor: str | None = None, dry_run: bool = False,
) -> list[str]:
    install_dir = install_dir.resolve()
    fps = metadata(name)["paper_fps"] if fps is None else fps
    input_path, gold_path, output_path = input_path.resolve(), gold_path.resolve(), output_path.resolve()
    inputs = validate_inputs(input_path, gold_path)
    verify_install(name, install_dir)
    if Path(checkpoint).exists():
        checkpoint = str(Path(checkpoint).resolve())
    if llm and Path(llm).exists():
        llm = str(Path(llm).resolve())
    if vision and Path(vision).exists():
        vision = str(Path(vision).resolve())
    if compressor:
        compressor = str(Path(compressor).resolve())
    if name == "dispider":
        config_path = Path(checkpoint) / "config.json"
        if not config_path.is_file():
            raise ValueError("Dispider requires a local legacy checkpoint directory; see external_baselines/README.md")
        architecture = json.loads(config_path.read_text(encoding="utf-8")).get("architectures", [])
        if "LongQwen2ForCausalLM" not in architecture:
            raise ValueError("Use the documented legacy Dispider checkpoint revision for this native implementation")
        compressor = compressor or str(Path(checkpoint) / "stream_compressor")
        if not (Path(compressor) / "config.json").is_file():
            raise FileNotFoundError(f"Missing Dispider stream compressor: {compressor}")
    work_dir = output_path.with_name(output_path.name + ".native")
    native_path = work_dir / "predictions.json"
    command, env, cwd = build_command(name, install_dir=install_dir, checkpoint=checkpoint,
        input_path=input_path, gold_path=gold_path, native_output=native_path,
        fps=fps, gpus=gpus, llm=llm, vision=vision, compressor=compressor)
    if dry_run:
        return command
    if output_path.exists() or work_dir.exists():
        raise FileExistsError(f"Choose a new output path: {output_path}")
    work_dir.mkdir(parents=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)
    raw_path = work_dir / metadata(name)["raw_directory"] / "predictions_raw_outputs.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    native_rows = json.loads(native_path.read_text(encoding="utf-8"))
    converted = convert_raw_outputs(name, inputs, raw_rows, native_rows)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
    temporary.replace(output_path)
    manifest = {"model_name": name, "protocol": "native_streaming", "fps": fps,
                "parser": metadata(name)["parser"], "upstream": metadata(name)["upstream"],
                "checkpoint": checkpoint, "llm_checkpoint": llm,
                "vision_checkpoint": vision or metadata(name).get("vision_checkpoint"), "compressor_checkpoint": compressor,
                "input_json": str(input_path), "gold_json": str(gold_path),
                "prediction_jsonl": str(output_path), "native_predictions": str(native_path),
                "raw_outputs": str(raw_path), "command": command}
    output_path.with_name(output_path.name + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return command
