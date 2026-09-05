#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT=${1:-dist/stream_betti_curves-source.zip}
mkdir -p "$(dirname "$OUTPUT")"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "package_release.sh expects a Git checkout so the archive matches tracked source exactly" >&2
  exit 2
fi

VERSION=$(awk -F'"' '/^version = / { print $2; exit }' Cargo.toml)
git archive --format=zip --prefix="stream_betti_curves-v${VERSION}/" --output="$OUTPUT" HEAD
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
echo "wrote $OUTPUT"
echo "wrote $OUTPUT.sha256"
