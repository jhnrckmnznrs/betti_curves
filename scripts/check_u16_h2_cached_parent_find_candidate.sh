#!/usr/bin/env bash
set -euo pipefail

STACK=${1:?usage: check_u16_h2_cached_parent_find_candidate.sh STACK [OUTPUT_CSV]}
OUTPUT=${2:-profiles/v23_3_u16_h2_cached_parent_find_paired.csv}
BINARY=${BINARY:-target/release/betti_curves}

python3 validation/check_u16_h2_cached_parent_find_equivalence.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26

mkdir -p "$(dirname "$OUTPUT")"
python3 scripts/profile_u16_h2_cached_parent_find.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26 \
  --warmup 1 \
  --repeats 11 \
  --diagnostic-repeats 1 \
  --start-with reference \
  --output "$OUTPUT"
