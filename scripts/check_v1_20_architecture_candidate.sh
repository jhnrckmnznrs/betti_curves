#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /path/to/CX09T1 [binary]" >&2
  exit 2
fi

STACK=$1
BIN=${2:-target/release/betti_curves}

if [[ ! -d "$STACK" ]]; then
  echo "stack directory not found: $STACK" >&2
  exit 2
fi
if [[ ! -x "$BIN" ]]; then
  echo "binary not found or not executable: $BIN" >&2
  exit 2
fi

python3 validation/check_scalar_h0_hierarchical_equivalence.py \
  "$STACK" --binary "$BIN" --slab-depths 8 16 32 --foreground-connectivity 26

python3 validation/check_scalar_h0_birth_buffer_equivalence.py \
  "$STACK" --binary "$BIN" --slice-limit 4 --slab-depth 2 --foreground-connectivity 26

python3 validation/check_scalar_h0_direct_event_equivalence.py \
  "$STACK" --binary "$BIN" --slice-limit 4 --slab-depth 2 --foreground-connectivity 26

python3 validation/check_scalar_h2_f32_end_to_end_equivalence.py \
  "$STACK" --binary "$BIN" --slice-limit 4 --slab-depth 2 --foreground-connectivity 26

python3 validation/check_scalar_merge_auto_equivalence.py \
  "$STACK" --binary "$BIN" --slab-depth 16 --foreground-connectivity 26

echo "PASS: v1.20 architecture candidate equivalence gate completed"
