#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 STACK_DIR [OUTPUT_CSV]" >&2
  exit 2
fi

STACK=$1
OUTPUT=${2:-profiles/v23_5_1_u16_h2_fast_interface_query_d16_confirmation.csv}
BINARY=target/release/betti_curves

# Re-run exact multi-depth equivalence before the focused performance confirmation.
python3 validation/check_u16_h2_fast_interface_query_equivalence.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26

mkdir -p "$(dirname "$OUTPUT")"

# Focused production-relevant confirmation: d16 only, 31 alternating A/B pairs,
# no sweep diagnostics/instrumentation.
python3 scripts/profile_u16_h2_fast_interface_query.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 \
  --foreground-connectivity 26 \
  --warmup 1 \
  --repeats 31 \
  --start-with reference \
  --output "$OUTPUT"

PAIRED_SUMMARY="${OUTPUT%.csv}_paired_summary.csv"
DECISION="${OUTPUT%.csv}_decision.txt"
python3 scripts/evaluate_u16_h2_fast_interface_query_confirmation.py \
  "$PAIRED_SUMMARY" \
  --output "$DECISION"
