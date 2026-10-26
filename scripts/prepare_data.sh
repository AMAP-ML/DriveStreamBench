#!/usr/bin/env bash
# Download the public annotation release and verify its pinned SHA-256.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "annotations" || "${1:-}" == "annotations/" ]]; then
  shift
fi
exec "${PYTHON:-python3}" "$REPO_ROOT/scripts/download_data.py" "$@"
