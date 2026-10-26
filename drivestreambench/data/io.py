"""JSONL / JSON I/O helpers — used everywhere annotations + predictions
flow between disk and Python objects.

Kept dependency-free; no pandas / pydantic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Union

PathLike = Union[str, Path]


def read_jsonl(path: PathLike) -> List[dict]:
    """Read all records from a JSONL file. Skips blank lines."""
    p = Path(path)
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iter_jsonl(path: PathLike) -> Iterator[dict]:
    """Stream records from a JSONL file (memory-efficient for large files)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: PathLike, records: Iterable[dict], *, ensure_ascii: bool = False) -> None:
    """Write records to a JSONL file, one object per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=ensure_ascii) + "\n")


def read_json(path: PathLike) -> Any:
    """Read a single JSON object (or array)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, obj: Any, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """Pretty-print a JSON object to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
