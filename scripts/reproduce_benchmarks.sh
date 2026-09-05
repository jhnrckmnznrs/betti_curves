#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=${PYTHON:-python3}
BINARY=${BETTI_BIN:-target/release/betti_curves}
WORK=${BETTI_REPRO_DIR:-/tmp/stream_betti_curves_repro}
THREADS=${BETTI_REPRO_THREADS:-"1 2 4"}

mkdir -p "$WORK"

cargo build --release --locked

"$PYTHON" benchmarks/generate_synthetic_stack.py \
  --output "$WORK/synthetic_64" \
  --shape 64 64 64 \
  --pattern blobs \
  --dtype u8 \
  --seed 42 \
  --overwrite

"$PYTHON" benchmarks/python_naive_baseline.py "$WORK/synthetic_64" \
  --foreground-connectivity 26 \
  --output-dir "$WORK/python_baseline"

# shellcheck disable=SC2086
"$PYTHON" benchmarks/run_suite.py "$WORK/synthetic_64" \
  --binary "$BINARY" \
  --modes h0-scalar-stream h2-scalar-stream \
  --slab-depths 4 8 16 \
  --threads $THREADS \
  --repeats 1 \
  --output "$WORK/benchmark_results.csv"

"$PYTHON" benchmarks/plot_results.py "$WORK/benchmark_results.csv" \
  --output-dir "$WORK/plots"

printf '\nReproducibility artifacts:\n  %s\n' "$WORK"
