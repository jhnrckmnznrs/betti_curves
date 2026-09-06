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

echo ">>> native U16 persistence-key model"
python3 validation/check_u16_native_key_model.py

echo ">>> U16 H2 plateau-zero persistence-elision model"
python3 validation/check_u16_h2_plateau_zero_elision_model.py --cases 5000

echo ">>> U16 H2 persistence root-dedup model"
python3 validation/check_u16_h2_root_dedup_persistence_model.py --cases 5000

echo ">>> U16 H2 two-hop parent-shortcut model"
python3 validation/check_u16_h2_two_hop_parent_shortcut_model.py --cases 5000

echo ">>> U16 H2 cached-parent find model"
python3 validation/check_u16_h2_cached_parent_find_model.py --cases 5000

echo ">>> U16 H2 fast interface-query arithmetic model"
python3 validation/check_u16_h2_fast_interface_query_model.py --cases 20000

echo ">>> compile public Python tooling (matches CI)"
python3 -m py_compile benchmarks/*.py validation/*.py scripts/*.py python_scripts/*.py

echo ">>> shell syntax"
for script in scripts/*.sh; do
    bash -n "$script"
done

echo ">>> cargo build --release"
cargo build --release --locked

echo "All local CI/pre-commit gates passed."
