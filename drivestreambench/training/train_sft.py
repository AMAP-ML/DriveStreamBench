"""DriveStream-SFT training launcher for ms-swift."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default hyperparameters reported in paper §5.5 / appendix.
DEFAULT_HPARAMS: Dict[str, Any] = {
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "model_type": "qwen3_vl",
    "batch_size": 1,
    "grad_accum": 4,
    "epochs": 3,
    "lr": 1e-5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "weight_decay": 0.1,
    "warmup_ratio": 0.03,
    "max_length": 40960,
    "image_max_tokens": 256,
    "frozen_vit": True,
    "frozen_aligner": True,
    "deepspeed": "configs/training/zero3.json",
    "use_liger_kernel": True,
    "save_steps": 250,
    "eval_steps": 250,
    "logging_steps": 10,
    "tuner_type": "full",  # full SFT (LoRA also supported by ms-swift)
    "seed": 42,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriveStream-SFT training launcher")
    # Data
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=False, default="")
    # Model / output
    p.add_argument("--model", default=DEFAULT_HPARAMS["model"])
    p.add_argument("--model-type", default=DEFAULT_HPARAMS["model_type"])
    p.add_argument("--output-dir", required=True)
    # Optimization
    p.add_argument("--batch-size", type=int, default=DEFAULT_HPARAMS["batch_size"])
    p.add_argument("--grad-accum", type=int, default=DEFAULT_HPARAMS["grad_accum"])
    p.add_argument("--epochs", type=int, default=DEFAULT_HPARAMS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["lr"])
    p.add_argument("--adam-beta1", type=float, default=DEFAULT_HPARAMS["adam_beta1"])
    p.add_argument("--adam-beta2", type=float, default=DEFAULT_HPARAMS["adam_beta2"])
    p.add_argument(
        "--weight-decay", type=float, default=DEFAULT_HPARAMS["weight_decay"]
    )
    p.add_argument(
        "--warmup-ratio", type=float, default=DEFAULT_HPARAMS["warmup_ratio"]
    )
    p.add_argument("--max-length", type=int, default=DEFAULT_HPARAMS["max_length"])
    p.add_argument(
        "--image-max-tokens", type=int, default=DEFAULT_HPARAMS["image_max_tokens"]
    )
    p.add_argument(
        "--frozen-vit", action="store_true", default=DEFAULT_HPARAMS["frozen_vit"]
    )
    p.add_argument(
        "--frozen-aligner",
        action="store_true",
        default=DEFAULT_HPARAMS["frozen_aligner"],
    )
    p.add_argument("--deepspeed", default=DEFAULT_HPARAMS["deepspeed"])
    p.add_argument(
        "--use-liger-kernel",
        action="store_true",
        default=DEFAULT_HPARAMS["use_liger_kernel"],
    )
    p.add_argument("--save-steps", type=int, default=DEFAULT_HPARAMS["save_steps"])
    p.add_argument("--eval-steps", type=int, default=DEFAULT_HPARAMS["eval_steps"])
    p.add_argument(
        "--logging-steps", type=int, default=DEFAULT_HPARAMS["logging_steps"]
    )
    p.add_argument(
        "--tuner-type", default=DEFAULT_HPARAMS["tuner_type"], choices=("full", "lora")
    )
    p.add_argument("--seed", type=int, default=DEFAULT_HPARAMS["seed"])
    # Backend
    p.add_argument(
        "--backend",
        default="ms-swift",
        choices=("ms-swift", "dry-run"),
        help="dry-run prints the resolved command then exits",
    )
    return p.parse_args(argv)


def build_command(args: argparse.Namespace) -> List[str]:
    """Resolve to the underlying backend CLI."""
    if args.backend in {"ms-swift", "dry-run"}:
        cmd = [
            "swift",
            "sft",
            "--model",
            args.model,
            "--model_type",
            args.model_type,
            "--dataset",
            args.train_jsonl,
            "--output_dir",
            args.output_dir,
            "--num_train_epochs",
            str(args.epochs),
            "--per_device_train_batch_size",
            str(args.batch_size),
            "--gradient_accumulation_steps",
            str(args.grad_accum),
            "--learning_rate",
            str(args.lr),
            "--adam_beta1",
            str(args.adam_beta1),
            "--adam_beta2",
            str(args.adam_beta2),
            "--weight_decay",
            str(args.weight_decay),
            "--torch_dtype",
            "bfloat16",
            "--warmup_ratio",
            str(args.warmup_ratio),
            "--max_length",
            str(args.max_length),
            "--save_steps",
            str(args.save_steps),
            "--logging_steps",
            str(args.logging_steps),
            "--seed",
            str(args.seed),
        ]
        if args.val_jsonl:
            cmd += [
                "--val_dataset",
                args.val_jsonl,
                "--eval_steps",
                str(args.eval_steps),
            ]
        if args.deepspeed:
            cmd += ["--deepspeed", args.deepspeed]
        if args.frozen_vit:
            cmd += ["--freeze_vit", "true"]
        if args.frozen_aligner:
            cmd += ["--freeze_aligner", "true"]
        if args.tuner_type == "lora":
            cmd += ["--tuner_type", "lora"]
        else:
            cmd += ["--tuner_type", "full"]
        if args.use_liger_kernel:
            cmd += ["--use_liger_kernel", "true"]
        return cmd

    raise ValueError(f"Unsupported training backend: {args.backend}")


def build_environment(args: argparse.Namespace) -> Dict[str, str]:
    """Forward the configured image-token budget to the model processor."""
    environment = dict(os.environ)
    environment["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_tokens)
    return environment


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_command(args)
    environment = build_environment(args)
    print(f"[train_sft] backend={args.backend}")
    print(f"[train_sft] IMAGE_MAX_TOKEN_NUM={environment['IMAGE_MAX_TOKEN_NUM']}")
    print(f"[train_sft] resolved command:\n  {' '.join(cmd)}")
    if args.backend == "dry-run":
        return
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).exists():
        print(
            f"\n[train_sft] ERROR: executable {cmd[0]!r} not found on PATH.\n"
            f"           Install with: 'pip install -e .[train]'\n"
            f"           (see drivestreambench/training/README.md).",
            file=sys.stderr,
        )
        sys.exit(1)
    rc = subprocess.call(cmd, env=environment)
    sys.exit(rc)


if __name__ == "__main__":
    main()
