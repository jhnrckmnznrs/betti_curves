# Contributing

Thank you for improving `stream_betti_curves`. The project treats correctness and performance as co-equal requirements: an optimization is not accepted if it changes the persistence multiset, and a memory optimization should be measured rather than inferred from data-structure size alone.

## Before opening a pull request

Run the repository gate from the project root:

```bash
scripts/prepare_for_commit.sh
```

The script runs `cargo fmt --all` first and then verifies formatting, tests, Clippy with warnings denied, and the locked release build. CI runs the verification steps again.


## Development setup

```bash
rustup show
cargo build --locked
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r validation/requirements.txt -r benchmarks/requirements.txt
```

## Required checks

Before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
python3 validation/test_oracles.py
```

For changes to scalar streaming or hierarchical persistence, also run the nearest dedicated equivalence gate under `validation/`.

## Performance changes

Include the following in a performance PR:

1. exact benchmark command;
2. dataset or synthetic-generator parameters;
3. Rust version and operating system;
4. `RAYON_NUM_THREADS`;
5. slab depth and foreground connectivity;
6. wall time and peak RSS;
7. scratch / filesystem I/O when relevant;
8. canonical persistence hash or exact multiset equivalence result;
9. before/after medians from at least three runs for small benchmarks.

Prefer changes that remove work or reduce state while preserving a simple invariant. Do not promote a heuristic based only on one specimen.

## Repository conventions

- Keep public documentation in `docs/`.
- Keep exploratory optimization notes in `docs/development/` rather than the repository root.
- Keep benchmark raw data under `benchmarks/data/` and generated figures under `benchmarks/plots/`.
- Do not commit large medical/research image volumes.
- Do not commit build outputs, temporary run files, generated persistence CSVs, or Python virtual environments.
- Avoid `unsafe` unless there is a measured need and the safety invariant is documented. The crate currently denies unsafe code except the isolated allocator-trim module.

## Commit and PR scope

Prefer small, auditable changes. For algorithmic work, a useful PR structure is:

1. reference path or retained oracle;
2. candidate implementation behind an explicit switch;
3. exactness gate;
4. profiler;
5. measured promotion decision;
6. cleanup after promotion.

## Reporting performance regressions

Use the performance-regression issue template. Include the command, input shape/type, slab depth, thread count, version/commit, peak RSS, wall time, and any `PROFILE_*` lines.
