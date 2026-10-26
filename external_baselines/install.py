#!/usr/bin/env python3
"""Obtain a pinned upstream source tree and apply DSB inference compatibility patches."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from external_baselines.native import BASELINES, ROOT, metadata, verify_install
from external_baselines.compatibility import configure_dispider_source


def install(name: str, install_dir: Path) -> None:
    meta = metadata(name)
    install_dir = install_dir.resolve()
    if not install_dir.exists():
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(install_dir)], check=True)
        subprocess.run(["git", "-C", str(install_dir), "remote", "add", "origin", meta["upstream"]["url"]], check=True)
        subprocess.run(["git", "-C", str(install_dir), "fetch", "--filter=blob:none", "--depth", "1", "origin", meta["upstream"]["commit"]], check=True)
        subprocess.run(["git", "-C", str(install_dir), "sparse-checkout", "set", "--cone", *meta["source_directories"]], check=True)
        subprocess.run(["git", "-C", str(install_dir), "checkout", "--detach", "FETCH_HEAD"], check=True)
    head = subprocess.check_output(["git", "-C", str(install_dir), "rev-parse", "HEAD"], text=True).strip()
    if head != meta["upstream"]["commit"]:
        raise ValueError(f"Use an empty installation directory or upstream revision {meta['upstream']['commit']}")
    patch = ROOT / "patches" / f"{name}.patch"
    if patch.exists():
        applied = subprocess.run(["git", "-C", str(install_dir), "apply", "--reverse", "--check", str(patch)], capture_output=True)
        if applied.returncode:
            subprocess.run(["git", "-C", str(install_dir), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(install_dir), "apply", str(patch)], check=True)
    if name == "dispider":
        configure_dispider_source(install_dir)
    verify_install(name, install_dir)
    print(f"{name}: {head}")
    print(f"Install the model environment and obtain weights as described at {meta['upstream']['installation_url']}")
    print("Run inference from that model's environment; source setup does not install Python packages or weights.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=BASELINES)
    parser.add_argument("--install-dir", type=Path, required=True)
    args = parser.parse_args()
    install(args.baseline, args.install_dir)


if __name__ == "__main__":
    main()
