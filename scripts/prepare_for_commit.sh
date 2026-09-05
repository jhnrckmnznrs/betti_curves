#!/usr/bin/env bash
set -euo pipefail

echo ">>> cargo fmt"
cargo fmt --all

echo ">>> cargo fmt --check"
cargo fmt --all -- --check

echo ">>> cargo test"
cargo test --all-targets --locked

echo ">>> cargo clippy"
cargo clippy --all-targets -- -D warnings

echo ">>> cargo build --release"
cargo build --release --locked

echo "All pre-commit Rust gates passed."
