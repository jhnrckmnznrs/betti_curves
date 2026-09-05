#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo test --all-targets
cargo clippy --all-targets -- -D warnings
python3 -m unittest discover -s python_scripts/tests -v
python3 -m unittest discover -s validation -p 'test_*.py' -v
python3 -m py_compile scripts/*.py validation/*.py
cargo build --release --locked

BETTI_CURVES_BINARY="${BETTI_CURVES_BINARY:-target/release/betti_curves}" \
  python3 validation/reference_check.py

STACK="${1:-${SCALAR_TEST_STACK:-${SCALAR_F32_TEST_STACK:-}}}"
if [[ -n "$STACK" ]]; then
  python3 validation/check_production_scalar_equivalence.py "$STACK" \
    --binary "$BETTI_CURVES_BINARY" \
    --slab-depths ${PRODUCTION_TEST_SLAB_DEPTHS:-8 16 32} \
    --foreground-connectivity "${SCALAR_TEST_CONNECTIVITY:-${SCALAR_F32_TEST_CONNECTIVITY:-26}}"
fi

echo "PASS: production release regression suite"
