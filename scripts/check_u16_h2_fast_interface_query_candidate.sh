#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 STACK_DIR [OUTPUT_CSV]" >&2
  exit 2
fi

STACK=$1
OUTPUT=${2:-profiles/v23_5_u16_h2_fast_interface_query_paired.csv}
BINARY=target/release/betti_curves

python3 validation/check_u16_h2_fast_interface_query_equivalence.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26

mkdir -p "$(dirname "$OUTPUT")"

python3 scripts/profile_u16_h2_fast_interface_query.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26 \
  --warmup 1 \
  --repeats 11 \
  --start-with reference \
  --output "$OUTPUT"
