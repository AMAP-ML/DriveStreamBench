"""Run L4 streaming evaluation: drive a session per annotation, write
prediction JSONL + manifest.

Console entry: ``python -m drivestreambench.inference.run_l4``

Flow per annotation:

  1. Load Sample from release-format JSONL
  2. SessionRunner.run(session_id=question_id, instruction=question, ...)
  3. Append session.to_prediction_record(model_name) to output JSONL

The model is *any* :class:`BaseModelAdapter`. For smoke testing, use
:class:`AlwaysSilent` or :class:`AlertOnTurn` from
:mod:`drivestreambench.models.dummy`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from drivestreambench.data.io import iter_jsonl, write_jsonl
from drivestreambench.data.schemas import SOURCE_FPS_DEFAULT, Sample
from drivestreambench.inference.manifest import new_manifest, write_manifest
from drivestreambench.models.base import BaseModelAdapter
from drivestreambench.protocol.frame_sampler import decision_count
from drivestreambench.protocol.runner import SessionRunner


def run_l4(
    *,
    annotations_path: str | Path,
    output_path: str | Path,
    model: BaseModelAdapter,
    fps: int = 2,
    frames_per_turn: int = 2,
    decision_cadence_hz: float = 1.0,
    source_fps: float = SOURCE_FPS_DEFAULT,
    n_decisions: Optional[int] = None,
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
    """Run inference on every L4 sample in ``annotations_path``.

    ``n_decisions`` defaults to all sampled frame groups, including a
    final partial group. Pass an explicit value to truncate the session.
    """
    runner = SessionRunner(
        model=model,
        fps=fps,
        frames_per_turn=frames_per_turn,
        decision_cadence_hz=decision_cadence_hz,
        source_fps=source_fps,
    )
    out_records: list[dict] = []
    n_seen = 0
    for raw in iter_jsonl(annotations_path):
        sample = Sample.from_dict(raw)
        if sample.question_level != "L4":
            continue
        n_seen += 1
        # Build full frame paths if ``dataroot`` provided
        frame_paths = (
            [str(Path(dataroot) / p) for p in sample.image_samples]
            if dataroot is not None
            else list(sample.image_samples)
        )
        n_total_frames = len(frame_paths)
        if not n_total_frames:
            raise ValueError(f"Sample {sample.question_id!r} has no frames")

        # Decide n_decisions for this session.
        if n_decisions is not None:
            n_t = n_decisions
        else:
            n_t = decision_count(
                n_total_frames,
                fps=fps,
                frames_per_turn=frames_per_turn,
                source_fps=source_fps,
            )

        sess = runner.run(
            session_id=sample.question_id,
            video_id=sample.video_name,
            instruction=sample.question,
            n_decisions=n_t,
            all_frame_paths=frame_paths,
            n_total_frames=n_total_frames,
        )
        out_records.append(sess.to_prediction_record(model_name=model.name))
        if progress and n_seen % 50 == 0:
            print(f"[run_l4] processed {n_seen} sessions", flush=True)

    write_jsonl(output_path, out_records)

    if manifest_path:
        manifest = new_manifest(
            benchmark_version=benchmark_version,
            split=split,
            model_name=model.name,
            protocol=model.protocol,
            fps=fps,
            frames_per_turn=frames_per_turn,
            decision_cadence_hz=decision_cadence_hz,
            framework=framework,
            framework_version=framework_version,
            hardware=hardware,
            precision=precision,
            prediction_jsonl=str(output_path),
        )
        write_manifest(manifest, manifest_path)

    if progress:
        print(
            f"[run_l4] wrote {len(out_records)} predictions → {output_path}", flush=True
        )


from drivestreambench.inference._model_loader import resolve_model as _resolve_model


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run L4 streaming inference + write predictions JSONL"
    )
    p.add_argument(
        "--annotations",
        required=True,
        help="path to L4 annotation JSONL (release schema)",
    )
    p.add_argument("--output", required=True, help="output predictions JSONL path")
    p.add_argument(
        "--model", default="dummy", help="model spec (dummy / alert_on_turn:N)"
    )
    p.add_argument(
        "--manifest", default="", help="optional path to write the eval manifest"
    )
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--frames-per-turn", type=int, default=2)
    p.add_argument("--cadence", type=float, default=1.0)
    p.add_argument("--source-fps", type=float, default=SOURCE_FPS_DEFAULT)
    p.add_argument("--n-decisions", type=int, default=None)
    p.add_argument(
        "--dataroot",
        default=None,
        help="prefix where {video_name}/CAM_F0/<file>.jpg lives (optional)",
    )
    p.add_argument("--split", default="")
    p.add_argument("--benchmark-version", default="")
    args = p.parse_args()

    model = _resolve_model(args.model)
    run_l4(
        annotations_path=args.annotations,
        output_path=args.output,
        model=model,
        fps=args.fps,
        frames_per_turn=args.frames_per_turn,
        decision_cadence_hz=args.cadence,
        source_fps=args.source_fps,
        n_decisions=args.n_decisions,
        dataroot=args.dataroot,
        manifest_path=args.manifest or None,
        split=args.split,
        benchmark_version=args.benchmark_version,
    )


if __name__ == "__main__":
    main()
