#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 STACK [OUTPUT_CSV]" >&2
  exit 2
fi

stack=$1
output=${2:-profiles/v23_u16_h2_root_dedup_paired.csv}

echo ">>> release build"
cargo build --release --locked

echo ">>> exact H2 root-dedup equivalence gate (d16/d32/d64)"
python3 validation/check_u16_h2_root_dedup_persistence_equivalence.py \
  "$stack" \
  --binary target/release/betti_curves \
  --slab-depths 16 32 64

echo ">>> interleaved paired H2 root-dedup profile (11 pairs/depth)"
python3 scripts/profile_u16_h2_root_dedup.py \
  "$stack" \
  --binary target/release/betti_curves \
  --slab-depths 16 32 64 \
  --warmup 1 \
  --repeats 11 \
  --output "$output"

echo ">>> candidate gate complete"
echo "raw:            $output"
echo "backend summary: ${output%.csv}_summary.csv"
echo "paired runs:     ${output%.csv}_pairs.csv"
echo "paired summary:  ${output%.csv}_paired_summary.csv"
