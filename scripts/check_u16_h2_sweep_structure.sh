#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 STACK [OUTPUT_CSV]" >&2
  exit 2
fi

STACK=$1
OUTPUT=${2:-profiles/v23_4_u16_h2_sweep_structure.csv}

mkdir -p "$(dirname "$OUTPUT")"
python3 scripts/profile_u16_h2_sweep_structure.py \
  "$STACK" \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 1 \
  --output "$OUTPUT"
