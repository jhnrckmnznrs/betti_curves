#!/usr/bin/env bash
set -euo pipefail
input=${1:-examples/CX09T1}
binary=${2:-target/release/betti_curves}
python3 validation/check_v1_24_h2_hierarchical_stream.py "$input" \
  --binary "$binary" --slice-limit 64 --slab-depths 4 8 16 --foreground-connectivity 26
