#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 STACK [BINARY]" >&2
  exit 2
fi

STACK=$1
BINARY=${2:-target/release/betti_curves}

echo ">>> v23.6 arithmetic oracle"
python3 validation/check_u16_h2_fast_interface_query_model.py --cases 20000

echo ">>> v23.6 exact OFF-vs-ON persistence equivalence (d16/d32/d64)"
python3 validation/check_u16_h2_fast_interface_query_equivalence.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26

echo ">>> v23.6 production-default / escape-hatch regression"
python3 validation/check_u16_h2_fast_interface_query_default.py \
  "$STACK" \
  --binary "$BINARY" \
  --slab-depth 16 \
  --foreground-connectivity 26

echo ">>> release binary version"
"$BINARY" --version

echo "PASS v23.6 U16 H2 fast-interface production freeze gate"
