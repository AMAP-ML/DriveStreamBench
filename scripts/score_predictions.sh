#!/usr/bin/env bash
# Score the bundled toy predictions and verify they match the bundled
# expected_outputs. This is the simplest user entry point — confirms
# the metric pipeline runs and produces the documented numbers.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "DriveStreamBench score_predictions"
echo "===================================="
echo

# L4 score
echo "[L4] Toy AlertScore"
python3 - <<'PYEOF'
import json
from drivestreambench.metrics.alertscore import score_l4_files

actual = score_l4_files(
    "examples/toy_annotations/L4_demo.jsonl",
    "examples/toy_predictions/L4_demo_predictions.jsonl",
    output_path="results/scores/L4_demo_scores.json",
)
expected = json.loads(open("examples/expected_outputs/L4_demo_scores.json").read())

agot = actual["primary_metric"]["value"]
aexp = expected["primary_metric"]["value"]
print(f"  Core Macro AlertScore@2s: {agot:.4f}  (expected {aexp:.4f})")
assert abs(agot - aexp) < 1e-6, f"L4 mismatch: {agot} vs {aexp}"
print("  L4 ✓")
print()
print("  Per subcategory:")
for cat, blk in actual["per_subcategory"].items():
    by2 = blk["by_tau"]["2.0"]
    print(f"    {cat:<24}  pos={blk['n_pos']} neg={blk['n_neg']}  "
          f"hit@2={by2['hit_rate']:.2f}  nsr={blk['nsr']:.2f}  AS@2={by2['alert_score']:.2f}")
PYEOF

echo

# L123 score
echo "[L1-L3] Toy reactive QA"
python3 - <<'PYEOF'
import json
from drivestreambench.metrics.l123_accuracy import score_l123_files

actual = score_l123_files(
    "examples/toy_annotations/L1L2L3_demo.jsonl",
    "examples/toy_predictions/L1L2L3_demo_predictions.jsonl",
    output_path="results/scores/L1L2L3_demo_scores.json",
)
expected = json.loads(open("examples/expected_outputs/L1L2L3_demo_scores.json").read())

agot = actual["summary"]["accuracy"]
aexp = expected["summary"]["accuracy"]
print(f"  Overall accuracy: {agot:.4f}  (expected {aexp:.4f})")
assert abs(agot - aexp) < 1e-6, f"L1-L3 mismatch: {agot} vs {aexp}"
print("  L1-L3 ✓")
print()
print("  Per level:")
for level, blk in actual["by_question_level"].items():
    print(f"    {level}: {blk['correct']}/{blk['total']} = {blk['accuracy']:.3f}")
PYEOF

echo
echo "===================================="
echo "Outputs written to:"
echo "  results/scores/L4_demo_scores.json"
echo "  results/scores/L1L2L3_demo_scores.json"
echo
echo "These match examples/expected_outputs/*.json."
echo "User-side metric reproducibility validated ✓"
