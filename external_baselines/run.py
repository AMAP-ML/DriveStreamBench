#!/usr/bin/env python3
"""Run an external model with its native DSB batch inference program."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from external_baselines.native import BASELINES, run_native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=BASELINES)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", type=Path, required=True, help="Native frame-input JSON")
    parser.add_argument("--gold", type=Path, required=True, help="Matching native gold JSON")
    parser.add_argument("--output", type=Path, required=True, help="Canonical prediction JSONL")
    parser.add_argument("--fps", type=int, choices=(1, 2), help="Defaults to the model's reported native FPS")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--llm", help="VideoLLM-online base language checkpoint")
    parser.add_argument("--vision", help="Vision checkpoint for Dispider, StreamForest, VideoLLM-online or M4")
    parser.add_argument("--compressor", help="Dispider stream_compressor directory, defaults to a child of --checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = run_native(args.baseline, install_dir=args.install_dir, checkpoint=args.checkpoint,
        input_path=args.input, gold_path=args.gold, output_path=args.output,
        fps=args.fps, gpus=args.gpus, llm=args.llm, vision=args.vision,
        compressor=args.compressor, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(command, indent=2))


if __name__ == "__main__":
    main()
