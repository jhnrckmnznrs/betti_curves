#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo test --all-targets
python3 -m unittest discover -s python_scripts/tests -v
python3 -m unittest discover -s validation -p 'test_*.py' -v
python3 -m py_compile \
  scripts/profile_scalar_stream.py \
  scripts/profile_scalar_stream_ablation.py \
  scripts/profile_scalar_f32_keys.py \
  scripts/profile_scalar_neighbor_kernel.py \
  scripts/profile_scalar_representative_active.py \
  scripts/profile_scalar_sweep_diagnostics.py \
  validation/check_scalar_stream_equivalence.py \
  validation/check_scalar_stream_ablation_equivalence.py \
  validation/check_scalar_f32_key_equivalence.py \
  validation/check_scalar_neighbor_kernel_equivalence.py \
  validation/check_scalar_representative_active_equivalence.py
cargo build --release --locked
BETTI_CURVES_BINARY="${BETTI_CURVES_BINARY:-target/release/betti_curves}" \
  python3 validation/reference_check.py

if [[ -n "${SCALAR_F32_TEST_STACK:-}" ]]; then
  python3 validation/check_scalar_f32_key_equivalence.py "$SCALAR_F32_TEST_STACK" \
    --binary "${BETTI_CURVES_BINARY:-target/release/betti_curves}" \
    --slab-depth "${SCALAR_F32_TEST_SLAB_DEPTH:-16}" \
    --foreground-connectivity "${SCALAR_F32_TEST_CONNECTIVITY:-26}"
  python3 validation/check_scalar_neighbor_kernel_equivalence.py "$SCALAR_F32_TEST_STACK" \
    --binary "${BETTI_CURVES_BINARY:-target/release/betti_curves}" \
    --slab-depth "${SCALAR_F32_TEST_SLAB_DEPTH:-16}" \
    --foreground-connectivity "${SCALAR_F32_TEST_CONNECTIVITY:-26}"
  python3 validation/check_scalar_representative_active_equivalence.py "$SCALAR_F32_TEST_STACK" \
    --binary "${BETTI_CURVES_BINARY:-target/release/betti_curves}" \
    --slab-depth "${SCALAR_F32_TEST_SLAB_DEPTH:-16}" \
    --foreground-connectivity "${SCALAR_F32_TEST_CONNECTIVITY:-26}"
fi

echo "PASS: optimized scalar-stream regression suite"
