"""Read and write evaluation settings, prediction paths and runtime metadata."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from drivestreambench.data.io import read_json, write_json
from drivestreambench.data.schemas import EvalManifest


MANIFEST_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Build / load / write
# ---------------------------------------------------------------------------

def new_manifest(
    *,
    benchmark_version: str,
    split: str,
    model_name: str,
    protocol: str = "native_streaming",
    parser: str = "strict_literal",
    fps: int = 2,
    frames_per_turn: int = 2,
    decision_cadence_hz: float = 1.0,
    temperature: float = 0.0,
    max_new_tokens: int = 8,
    framework: str = "",
    framework_version: str = "",
    hardware: str = "",
    precision: str = "bf16",
    model_checkpoint: str = "",
    model_checkpoint_hash: str = "",
    prediction_jsonl: str = "",
    score_json: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> EvalManifest:
    """Construct a new EvalManifest with sane defaults."""
    return EvalManifest(
        benchmark="DriveStreamBench",
        benchmark_version=benchmark_version,
        split=split,
        model_name=model_name,
        model_checkpoint=model_checkpoint,
        model_checkpoint_hash=model_checkpoint_hash,
        protocol=protocol,  # type: ignore[arg-type]
        parser=parser,  # type: ignore[arg-type]
        decision_cadence_hz=decision_cadence_hz,
        fps=fps,
        frames_per_turn=frames_per_turn,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        framework=framework,
        framework_version=framework_version,
        hardware=hardware,
        precision=precision,
        prediction_jsonl=prediction_jsonl,
        score_json=score_json,
        extra=dict(extra or {}),
    )


def write_manifest(manifest: EvalManifest, path: str | os.PathLike) -> None:
    """Serialize the manifest to a JSON file with environment metadata."""
    payload: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": _safe_hostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **dataclasses.asdict(manifest),
    }
    write_json(path, payload)


def read_manifest(path: str | os.PathLike) -> EvalManifest:
    """Load a manifest JSON from disk into an :class:`EvalManifest`."""
    payload = read_json(path)
    fields = {f.name for f in dataclasses.fields(EvalManifest)}
    keep = {k: v for k, v in payload.items() if k in fields}
    keep.setdefault("extra", {})
    return EvalManifest(**keep)


def _safe_hostname() -> str:
    try:
        h = socket.gethostname()
    except Exception:
        return ""

    if "." in h:
        h = h.split(".", 1)[0]
    return h


# ---------------------------------------------------------------------------
# Hashing helpers (for reproducibility)
# ---------------------------------------------------------------------------

def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file. Streams chunks for large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | os.PathLike) -> str:
    """SHA-256 of a directory tree (sorted by relative path)."""
    p = Path(path)
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(p)).encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(sha256_file(f).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


__all__ = [
    "MANIFEST_VERSION",
    "EvalManifest",
    "new_manifest",
    "write_manifest",
    "read_manifest",
    "sha256_file",
    "sha256_dir",
]
