#!/usr/bin/env bash
set -euo pipefail

echo ">>> cargo fmt"
cargo fmt --all

echo ">>> cargo fmt --check (matches CI)"
cargo fmt --all -- --check

echo ">>> cargo clippy (matches CI)"
cargo clippy --all-targets -- -D warnings

echo ">>> cargo test (matches CI)"
cargo test --all-targets --locked

echo ">>> independent Python oracles (matches CI)"
python3 validation/test_oracles.py

echo ">>> compile public Python tooling (matches CI)"
python3 -m py_compile benchmarks/*.py validation/*.py scripts/*.py python_scripts/*.py

echo ">>> cargo build --release"
cargo build --release --locked

echo "All local CI/pre-commit gates passed."
