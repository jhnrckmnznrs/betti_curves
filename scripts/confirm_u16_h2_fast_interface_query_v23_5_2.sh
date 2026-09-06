#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 STACK_DIR [OUTPUT_CSV]" >&2
  exit 2
fi

STACK=$1
OUTPUT=${2:-profiles/v23_5_2_u16_h2_fast_interface_query_d16_blocked.csv}
BINARY=target/release/betti_curves

python3 validation/check_u16_h2_fast_interface_query_equivalence.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26

mkdir -p "$(dirname "$OUTPUT")"

python3 scripts/profile_u16_h2_fast_interface_query_blocked.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --blocks 5 \
  --pairs-per-block 7 \
  --warmup-pairs-per-block 1 \
  --cooldown-seconds 10 \
  --start-with reference \
  --perf auto \
  --output "$OUTPUT"

SUMMARY="${OUTPUT%.csv}_summary.csv"
DECISION="${OUTPUT%.csv}_decision.txt"
python3 scripts/evaluate_u16_h2_fast_interface_query_blocked.py \
  "$SUMMARY" \
  --output "$DECISION"
