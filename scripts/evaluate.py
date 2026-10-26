#!/usr/bin/env python3
"""Score model predictions against DriveStreamBench annotations."""

from __future__ import annotations

import argparse
import json

from drivestreambench.data.io import read_jsonl, write_json
from drivestreambench.metrics.alertscore import score_l4_files
from drivestreambench.metrics.l3_event import binary_f1_l3
from drivestreambench.metrics.l123_accuracy import score_l123_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["l4", "l123"], default="l4")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--parser", choices=["strict_literal", "action_level"], default="strict_literal"
    )
    parser.add_argument(
        "--fps",
        type=int,
        choices=[1, 2],
        default=2,
        help="Model input FPS for raw source-frame references (default: 2)",
    )
    args = parser.parse_args()
    if args.task == "l4":
        result = score_l4_files(
            args.annotations,
            args.predictions,
            output_path=args.output,
            parser_mode=args.parser,
            fps=args.fps,
        )
        print(json.dumps(result["primary_metric"], indent=2))
    else:
        result = score_l123_files(
            args.annotations, args.predictions, output_path=args.output
        )
        event = binary_f1_l3(read_jsonl(args.annotations), read_jsonl(args.predictions))
        result["l3_event"] = event
        levels = result["by_question_level"]
        if all(level in levels for level in ["L1", "L2", "L3"]):
            result["primary_metric"] = {
                "name": "L1-L3 Macro",
                "value": (
                    levels["L1"]["accuracy"] + levels["L2"]["accuracy"] + event["f1"]
                )
                / 3,
            }
        write_json(args.output, result)
        print(
            json.dumps(
                result.get("primary_metric", result["by_question_level"]), indent=2
            )
        )


if __name__ == "__main__":
    main()
