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

echo ">>> compact leaf-state equivalence model"
python3 validation/check_compact_leaf_state_model.py --cases 5000

echo ">>> leaf-history watch-filter equivalence model"
python3 validation/check_leaf_history_watch_filter_model.py --cases 5000

echo ">>> plateau-native leaf zero-death elision model"
python3 validation/check_plateau_native_leaf_elision_model.py --cases 5000

echo ">>> H2 shell/root pruning equivalence model"
python3 validation/check_h2_shell_root_pruning_model.py --cases 5000

echo ">>> compile public Python tooling (matches CI)"
python3 -m py_compile benchmarks/*.py validation/*.py scripts/*.py python_scripts/*.py

echo ">>> cargo build --release"
cargo build --release --locked

echo "All local CI/pre-commit gates passed."
