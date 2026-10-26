#!/usr/bin/env bash
# Smoke test — validates the full DriveStreamBench harness end-to-end on
# bundled toy data (CPU only, <30 s on a laptop). Users should run
# this immediately after install to confirm everything is wired correctly.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pass()  { printf "\033[32m  ✓\033[0m %s\n" "$1"; }
fail()  { printf "\033[31m  ✗\033[0m %s\n" "$1"; exit 1; }
info()  { printf "  → %s\n" "$1"; }

echo "DriveStreamBench smoke test"
echo "============================="
echo

# 1. Check Python ≥ 3.9
echo "[1] Python version"
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "python3 = $PYVER"
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ required"' || fail "Python ≥ 3.9 required"
pass "Python $PYVER OK"

# 2. Package importable (CPU-only path)
echo
echo "[2] Package imports"
python3 -c 'from drivestreambench import score_l4, score_l123, L4Sample' || fail "package import failed"
pass "drivestreambench imports OK"

# 3. Validate bundled toy annotations
echo
echo "[3] Toy annotations validate against release schema"
python3 -m drivestreambench.data.validate \
  examples/toy_annotations/L4_demo.jsonl >/dev/null || fail "L4 toy annotations failed validation"
pass "L4_demo.jsonl validates"
python3 -m drivestreambench.data.validate \
  examples/toy_annotations/L1L2L3_demo.jsonl >/dev/null || fail "L1L2L3 toy annotations failed validation"
pass "L1L2L3_demo.jsonl validates"

# 4. Run dummy AlertOnTurn through L4 toy session (full protocol path)
echo
echo "[4] Run dummy model through L4 protocol"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python3 -m drivestreambench.inference.run_l4 \
  --annotations examples/toy_annotations/L4_demo.jsonl \
  --output "$TMP/pred.jsonl" \
  --model alert_on_turn:4 \
  --manifest "$TMP/manifest.json" 2>&1 >/dev/null \
  || fail "run_l4 failed"
[ -s "$TMP/pred.jsonl" ] || fail "run_l4 produced empty predictions"
[ -s "$TMP/manifest.json" ] || fail "run_l4 did not write manifest"
pass "run_l4 produced predictions + manifest"

# 5. Score the predictions and check primary metric
echo
echo "[5] Score predictions"
python3 - "$TMP/pred.jsonl" <<'PYEOF' || fail "scoring failed"
import json, sys
from drivestreambench.metrics.alertscore import score_l4_files
result = score_l4_files(
    "examples/toy_annotations/L4_demo.jsonl",
    sys.argv[1],
)
expected = json.loads(open("examples/expected_outputs/L4_demo_scores.json").read())
got_v = result["primary_metric"]["value"]
exp_v = expected["primary_metric"]["value"]
assert abs(got_v - exp_v) < 1e-6, f"score mismatch: got {got_v}, expected {exp_v}"
print(f"  primary metric: {result['primary_metric']['name']} = {got_v:.4f}")
PYEOF
pass "score matches examples/expected_outputs/L4_demo_scores.json"

# 6. Run pytest if available
echo
echo "[6] Run pytest suite (optional — install with 'pip install pytest')"
if command -v pytest >/dev/null 2>&1 || python3 -c 'import pytest' 2>/dev/null; then
  python3 -m pytest tests/ -q 2>&1 | tail -5
  pass "pytest suite passed"
else
  info "pytest not installed; skipping (but install via 'pip install -e .[test]' to run)"
fi

echo
echo "============================="
echo "Smoke test PASSED"
echo "Next step: bash scripts/score_predictions.sh"
