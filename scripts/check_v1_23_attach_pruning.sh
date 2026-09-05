#!/usr/bin/env bash
set -euo pipefail
INPUT=${1:-examples/CX09T1}
BINARY=${2:-target/release/betti_curves}
python3 validation/check_scalar_h0_hierarchical_attach_pruning_equivalence.py \
  "$INPUT" --binary "$BINARY" --slice-limit 64 --slab-depths 4 8 16 --foreground-connectivity 26
