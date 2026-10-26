#!/usr/bin/env python3
"""Download, verify and install the versioned DriveStreamBench annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("Invalid archive path: " + name)
    for prefix, dest in [
        ("DriveStreamBench_Core/annotations/", "annotations/"),
        ("DriveStreamBench_OOD/annotations/", "annotations/"),
        ("DriveStreamBench_OOD/code/", "external/ood/code/"),
        ("DriveStreamBench_OOD/file/", "external/ood/file/"),
    ]:
        if name.startswith(prefix):
            return dest + name[len(prefix) :]
    return {
        "DriveStreamBench_Core/schema.json": "manifests/release_schema.json",
        "DriveStreamBench_OOD/OOD_data_process.md": "external/ood/OOD_data_process.md",
        "split_index.csv": "manifests/split_index.csv",
        "README.md": "docs/dataset.md",
        "LICENSE-DATA.txt": "external/ood/LICENSE-DATA.txt",
    }.get(name)


def install(archive, root, manifest):
    if sha256(archive) != manifest["archive_sha256"]:
        raise RuntimeError("Archive checksum mismatch; no files installed.")
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise RuntimeError("Archive integrity check failed.")
        for item in manifest["annotation_files"]:
            if (
                hashlib.sha256(z.read(item["archive_path"])).hexdigest()
                != item["sha256"]
            ):
                raise RuntimeError("Annotation checksum mismatch: " + item["filename"])
        plans = []
        for item in z.infolist():
            if item.is_dir():
                continue
            relative = target_path(item.filename)
            if relative is None:
                continue
            target = (root / relative).resolve()
            target.relative_to(root)
            content = z.read(item)
            if target.exists() and target.read_bytes() != content:
                raise RuntimeError("Conflicting local file: " + str(target))
            plans.append((target, content))
        for target, content in plans:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    for item in manifest["annotation_files"]:
        if sha256(root / "annotations" / item["filename"]) != item["sha256"]:
            raise RuntimeError(
                "Installed annotation checksum mismatch: " + item["filename"]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Install a previously downloaded ZIP."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1] / "manifests/data_release.json"
        ).read_text()
    )
    archive = args.archive or (
        (args.cache_dir or args.repo_root / ".cache/dsb") / manifest["archive_filename"]
    )
    if not archive.is_file():
        if args.archive:
            parser.error("Archive not found: " + str(archive))
        archive.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="dsb-", suffix=".part", dir=archive.parent)
        os.close(fd)
        temp = Path(name)
        try:
            request = urllib.request.Request(
                manifest["archive_url"],
                headers={"User-Agent": "DriveStreamBench/1.0.0"},
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temp.open("wb") as output,
            ):
                while chunk := response.read(1 << 20):
                    output.write(chunk)
            if sha256(temp) != manifest["archive_sha256"]:
                raise RuntimeError("Downloaded archive checksum mismatch.")
            temp.replace(archive)
        finally:
            temp.unlink(missing_ok=True)
    try:
        install(archive, args.repo_root, manifest)
    except (RuntimeError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
    print(
        "Verified "
        + str(len(manifest["annotation_files"]))
        + " annotation files, "
        + str(sum(item["row_count"] for item in manifest["annotation_files"]))
        + " records."
    )


if __name__ == "__main__":
    main()
