# Validation strategy

The validation policy is simple: **performance changes must preserve the exact topological result**.

## Layers of checking

### Rust unit tests

Unit tests cover parsers, union-find invariants, scalar ordering, memory-layout switches, and specialized reducer behavior.

```bash
cargo test --all-targets
```

### CLI integration tests

`tests/cli_integration.rs` runs the compiled executable on committed tiny TIFF fixtures. It checks help/version behavior, a known H0 essential interval, a known H2 cavity interval, and equivalence between flat and hierarchical scalar modes on a tiny F32 stack.

### Independent Python oracles

`validation/tiny_cubical_oracle.py` builds cubical boundary matrices over F2 for tiny images. It is intentionally independent of the production union-find implementation.

`validation/test_oracles.py` includes exhaustive 2×2×2 checks plus a shell with nontrivial H2.

### Equivalence gates

The `validation/check_*.py` scripts compare reference and candidate paths. Persistence outputs are treated as **multisets**; row order is not a correctness condition.

Canonical comparisons therefore sort `(birth, death, multiplicity)` rather than hashing emission order.

## Recommended release gate

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
python3 -m pip install -r validation/requirements.txt
python3 validation/test_oracles.py
```

For changes to H0/H2 hierarchical code, also run the corresponding focused validation script on a representative F32 volume.

## Numerical scope

Scalar modes preserve exact stored-value ordering; they do not resample, normalize, or reinterpret TIFF photometric metadata. Validation therefore checks topology of the decoded scalar stack, not upstream acquisition/calibration semantics.
