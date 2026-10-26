"""Tests for :mod:`drivestreambench.inference.manifest`."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from drivestreambench.data.schemas import EvalManifest
from drivestreambench.inference.manifest import (
    MANIFEST_VERSION,
    new_manifest,
    read_manifest,
    sha256_file,
    write_manifest,
)


def test_new_manifest_defaults():
    m = new_manifest(
        benchmark_version="1.0.0",
        split="l4_core_test",
        model_name="dummy",
    )
    assert isinstance(m, EvalManifest)
    assert m.benchmark == "DriveStreamBench"
    assert m.benchmark_version == "1.0.0"
    assert m.protocol in ("native_streaming", "mode_alpha")
    assert m.fps == 2
    assert m.frames_per_turn == 2
    assert m.decision_cadence_hz == 1.0


def test_manifest_roundtrip(tmp_path: Path):
    m = new_manifest(
        benchmark_version="1.0.0",
        split="l4_core_test",
        model_name="qwen3vl_8b",
        protocol="native_streaming",
        fps=2,
        frames_per_turn=2,
        framework="vllm",
        framework_version="0.10",
        hardware="1xL20-48G",
        prediction_jsonl="results/predictions/qwen3vl_8b_l4.jsonl",
    )
    p = tmp_path / "manifest.json"
    write_manifest(m, p)
    payload = json.loads(p.read_text())
    assert payload["manifest_version"] == MANIFEST_VERSION
    assert "created_at_utc" in payload
    assert payload["model_name"] == "qwen3vl_8b"

    loaded = read_manifest(p)
    assert loaded.model_name == m.model_name
    assert loaded.protocol == m.protocol
    assert loaded.framework == m.framework
    assert loaded.prediction_jsonl == m.prediction_jsonl


def test_sha256_file_deterministic(tmp_path: Path):
    content = b"DriveStreamBench reproducibility check\n"
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(content)
    f2.write_bytes(content)
    assert sha256_file(f1) == sha256_file(f2)
    assert len(sha256_file(f1)) == 64    # hex digest


def test_new_manifest_default_parser_strict():
    m = new_manifest(
        benchmark_version="1.0.0",
        split="l4_core_test",
        model_name="qwen3vl_8b",
    )
    assert m.parser == "strict_literal"


def test_new_manifest_action_level_parser():
    m = new_manifest(
        benchmark_version="1.0.0",
        split="l4_core_test",
        model_name="dispider",
        parser="action_level",
    )
    assert m.parser == "action_level"


def test_manifest_parser_field_roundtrips(tmp_path: Path):
    m = new_manifest(
        benchmark_version="1.0.0",
        split="l4_core_test",
        model_name="streamforest",
        parser="action_level",
    )
    p = tmp_path / "manifest.json"
    write_manifest(m, p)
    loaded = read_manifest(p)
    assert loaded.parser == "action_level"
