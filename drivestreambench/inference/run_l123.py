"""Run L1-L3 reactive QA inference: load annotations, ask each model
adapter for a single answer per question, write prediction JSONL in the
release ``output_schema.json`` format::

    {"question_id": str, "prediction": str, "confidence": float | null}

Console entry: ``python -m drivestreambench.inference.run_l123``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from drivestreambench.data.io import iter_jsonl, write_jsonl
from drivestreambench.data.schemas import SOURCE_FPS_DEFAULT, Sample
from drivestreambench.inference.manifest import new_manifest, write_manifest
from drivestreambench.models.base import BaseModelAdapter
from drivestreambench.protocol.prompts import (
    L1L2_MCQ_SYSTEM_PROMPT,
    L1L2L3_BINARY_SYSTEM_PROMPT,
    build_l1l2l3_user_message,
)


def run_l123(
    *,
    annotations_path: str | Path,
    output_path: str | Path,
    model: BaseModelAdapter,
    dataroot: Optional[str] = None,
    manifest_path: Optional[str | Path] = None,
    benchmark_version: str = "",
    split: str = "",
    framework: str = "",
    framework_version: str = "",
    hardware: str = "",
    precision: str = "bf16",
    progress: bool = True,
) -> None:
    """Run reactive QA inference; write prediction JSONL."""
    out_records: list[dict] = []
    n_seen = 0
    for raw in iter_jsonl(annotations_path):
        sample = Sample.from_dict(raw)
        if sample.question_level not in ("L1", "L2", "L3"):
            continue
        n_seen += 1
        frame_paths = (
            [str(Path(dataroot) / p) for p in sample.image_samples]
            if dataroot is not None
            else list(sample.image_samples)
        )
        # MCQ samples in release have answer "A"-"D" but options are
        # baked into the question text — pass options=None.
        user_msg = build_l1l2l3_user_message(sample.question, options=None)
        original_system_prompt = getattr(model, "system_prompt", None)
        if original_system_prompt is not None:
            model.system_prompt = (
                L1L2_MCQ_SYSTEM_PROMPT
                if sample.question_format == "mcq"
                else L1L2L3_BINARY_SYSTEM_PROMPT
            )
        try:
            if hasattr(model, "reset_session"):
                model.reset_session(sample.question_id)
            reply = model.step(
                frames=frame_paths,
                instruction=user_msg,
                history=[],
                video_time=sample.video_duration_seconds(SOURCE_FPS_DEFAULT) or 0.0,
            )
        finally:
            if original_system_prompt is not None:
                model.system_prompt = original_system_prompt
        out_records.append(
            {
                "question_id": sample.question_id,
                "prediction": reply,
            }
        )
        if progress and n_seen % 200 == 0:
            print(f"[run_l123] processed {n_seen} questions", flush=True)

    write_jsonl(output_path, out_records)

    if manifest_path:
        manifest = new_manifest(
            benchmark_version=benchmark_version,
            split=split,
            model_name=model.name,
            protocol=model.protocol,
            framework=framework,
            framework_version=framework_version,
            hardware=hardware,
            precision=precision,
            prediction_jsonl=str(output_path),
        )
        write_manifest(manifest, manifest_path)

    if progress:
        print(
            f"[run_l123] wrote {len(out_records)} predictions → {output_path}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Run L1-L3 reactive QA inference")
    p.add_argument(
        "--annotations",
        required=True,
        help="release JSONL with L1/L2/L3 records (combined or single-level)",
    )
    p.add_argument("--output", required=True, help="output predictions JSONL path")
    p.add_argument(
        "--model",
        default="dummy_no",
        help="model spec — see drivestreambench.inference._model_loader. "
        "Examples: 'dummy_yes' / 'dummy_no' / 'api_vlm' / 'qwen3vl:<path>'",
    )
    p.add_argument(
        "--manifest", default="", help="optional path to write the eval manifest"
    )
    p.add_argument(
        "--dataroot",
        default=None,
        help="prefix where {video_name}/CAM_F0/<file>.jpg lives (optional)",
    )
    p.add_argument("--split", default="")
    p.add_argument("--benchmark-version", default="")
    args = p.parse_args()

    from drivestreambench.inference._model_loader import resolve_model

    model = resolve_model(args.model)

    run_l123(
        annotations_path=args.annotations,
        output_path=args.output,
        model=model,
        dataroot=args.dataroot,
        manifest_path=args.manifest or None,
        split=args.split,
        benchmark_version=args.benchmark_version,
    )


if __name__ == "__main__":
    main()
