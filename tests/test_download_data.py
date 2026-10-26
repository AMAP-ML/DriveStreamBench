"""Dataset installation integrity and filesystem-boundary checks."""

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "dsb_data_download",
    Path(__file__).resolve().parents[1] / "scripts/download_data.py",
)
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)
ANNOTATION = b'{"question_id":"sample"}\n'
MEMBER = "DriveStreamBench_Core/annotations/l4_core_test.jsonl"


def bundle(tmp_path, extra=None):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(MEMBER, ANNOTATION)
        z.writestr("DriveStreamBench_Core/schema.json", "{}")
        if extra:
            z.writestr(extra, b"content")
    manifest = {
        "archive_sha256": DOWNLOAD.sha256(archive),
        "annotation_files": [
            {
                "filename": "l4_core_test.jsonl",
                "archive_path": MEMBER,
                "sha256": hashlib.sha256(ANNOTATION).hexdigest(),
            }
        ],
    }
    return archive, manifest


def test_install_and_repeat_preserve_annotation(tmp_path):
    archive, manifest = bundle(tmp_path)
    root = tmp_path / "repo"
    DOWNLOAD.install(archive, root, manifest)
    DOWNLOAD.install(archive, root, manifest)
    assert (root / "annotations/l4_core_test.jsonl").read_bytes() == ANNOTATION
    assert (root / "manifests/release_schema.json").read_text() == "{}"


def test_corrupt_archive_rejected_before_install(tmp_path):
    archive, manifest = bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"x")
    root = tmp_path / "repo"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        DOWNLOAD.install(archive, root, manifest)
    assert not root.exists()


def test_conflicting_local_annotation_preserved(tmp_path):
    archive, manifest = bundle(tmp_path)
    root = tmp_path / "repo"
    existing = root / "annotations/l4_core_test.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"local annotation")
    with pytest.raises(RuntimeError, match="Conflicting local file"):
        DOWNLOAD.install(archive, root, manifest)
    assert existing.read_bytes() == b"local annotation"
    assert not (root / "manifests/release_schema.json").exists()


@pytest.mark.parametrize(
    "member",
    [
        "../escape",
        "/absolute/path",
        "DriveStreamBench_Core/annotations/../../../escape",
    ],
)
def test_archive_path_boundary_is_enforced_before_writes(tmp_path, member):
    archive, manifest = bundle(tmp_path, member)
    root = tmp_path / "repo"
    with pytest.raises(RuntimeError, match="Invalid archive path"):
        DOWNLOAD.install(archive, root, manifest)
    assert not root.exists()
