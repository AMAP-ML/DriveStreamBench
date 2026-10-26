#!/usr/bin/env bash
# Build the short-version L4 annotation subset from a full L4-Core test
# annotation file. Wraps drivestreambench.data.build_short with the
# paper's default crop window (15 seconds @ 10 FPS source = 150 frames).
#
# This is the input that closed-source API VLMs (GPT-5.4, Gemini 3.1 Pro)
# are evaluated on — they cannot persist a KV cache across calls and must
# re-encode the full visible history every turn, so cropping keeps the
# per-call cost bounded.
#
# Usage:
#
#   bash scripts/build_l4_short.sh \
#     [INPUT=annotations/l4_core_test.jsonl] \
#     [OUTPUT=annotations/l4_short_test.jsonl] \
#     [DSB_SHORT_SECONDS=15] \
#     [SOURCE_FPS=10]
#
# All four are env-var overridable; the defaults match paper §6.

set -euo pipefail

INPUT="${INPUT:-annotations/l4_core_test.jsonl}"
OUTPUT="${OUTPUT:-annotations/l4_short_test.jsonl}"
SECONDS_VAL="${DSB_SHORT_SECONDS:-15}"
SOURCE_FPS_VAL="${SOURCE_FPS:-10}"

if [ ! -f "$INPUT" ]; then
  cat <<EOF >&2
ERROR: input annotation not found: $INPUT

Run scripts/prepare_data.sh annotations/ first to fetch
the full L4-Core test annotation pack from the versioned release.
EOF
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

python3 -m drivestreambench.data.build_short \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --seconds "$SECONDS_VAL" \
    --source-fps "$SOURCE_FPS_VAL"

echo
echo "[build_l4_short] wrote $OUTPUT"
echo "[build_l4_short] downstream usage:"
echo "  python -m drivestreambench.inference.run_l4 \\"
echo "      --annotations $OUTPUT \\"
echo "      --output      results/predictions/<your_model>_l4_short_test.jsonl \\"
echo "      --model       <your_model_alias>"
