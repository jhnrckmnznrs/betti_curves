#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 STACK SLAB_DEPTH FOREGROUND_CONNECTIVITY MODE [extra betti_curves args...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STACK=$1
SLAB=$2
CONNECTIVITY=$3
MODE=$4
shift 4

if ! command -v cargo-flamegraph >/dev/null 2>&1 && ! cargo flamegraph --help >/dev/null 2>&1; then
  echo "cargo-flamegraph is required: cargo install flamegraph" >&2
  exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR=${BETTI_PROFILE_DIR:-profiles/$STAMP}
mkdir -p "$OUTDIR"
META="$OUTDIR/metadata.txt"
SVG="$OUTDIR/flamegraph.svg"

{
  echo "utc=$STAMP"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "uname=$(uname -a)"
  echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-default}"
  printf 'command=target/release/betti_curves %q %q %q %q' "$STACK" "$SLAB" "$CONNECTIVITY" "$MODE"
  printf ' %q' "$@"
  printf '\n'
} > "$META"

cargo build --release --locked
cargo flamegraph --release --bin betti_curves --output "$SVG" -- \
  "$STACK" "$SLAB" "$CONNECTIVITY" "$MODE" "$@"

echo "wrote $SVG"
echo "wrote $META"
