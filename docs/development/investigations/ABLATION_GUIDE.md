# Scalar persistence stream ablation guide

The v1.1 experiment identified the production algorithmic base; v1.2 keeps
that ablation machinery and adds a separately measurable native-F32
representation optimization.

## 1. Build and regression check

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release --locked
scripts/check_optimized_scalar_stream.sh
```

## 2. Independent correctness check

The measured production defaults are `scan + radix + verify + native32`:

```bash
python3 validation/check_scalar_stream_equivalence.py examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26
```

The original three implementation choices can also be checked explicitly:

```bash
python3 validation/check_scalar_stream_equivalence.py examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26 \
  --merge-strategy scan \
  --interface-order comparison \
  --event-order resort
```

Both commands must report exact H0/H2 interval-multiset agreement with the
in-memory scalar implementation.

For the strongest one-time check, compare all eight stream cells with the
in-memory oracle:

```bash
python3 validation/check_scalar_stream_ablation_equivalence.py examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26
```

Use `--quick` to check only the legacy and measured production-base cells.

## 3. Full factorial performance experiment

```bash
python3 scripts/profile_scalar_stream_ablation.py examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26 \
  --repeats 3 \
  --output cx09t1_ablation_d16.csv
```

The eight cells are:

| Merge | Interface order | Event order |
|---|---|---|
| scan | comparison | resort |
| scan | comparison | verify |
| scan | radix | resort |
| scan | radix | verify |
| heap | comparison | resort |
| heap | comparison | verify |
| heap | radix | resort |
| heap | radix | verify |

`legacy` denotes `scan + comparison + resort`; `production-base` denotes
`scan + radix + verify`. The F32 key mode is held fixed by the profiler and is
`native32` by default.

The profiler records the execution sequence explicitly, rotates/reverses cell
order between repeats, and verifies one canonical persistence hash per homology
degree before reporting a winner.

## 4. Decision rule for the next patch

Do not select a strategy from pooled factor medians alone. Use the individual
factorial cells and H0/H2 preparation/reduction timings.

- If `scan` beats `heap` consistently, restore scan as the default until a
  larger-run crossover is demonstrated.
- If `radix` wins during preparation, keep it; if not, retain comparison order
  for current 64-bit scalar keys and revisit radix together with native F32.
- If `verify` wins, keep the monotonicity invariant and remove the redundant
  sorts permanently. If the difference is negligible, `verify` is still
  preferable because it avoids unnecessary work while checking correctness.
The CX09T1 slab-depth-16 experiment selected `scan + radix + verify`. For F32
representation profiling, run:

```bash
python3 validation/check_scalar_f32_key_equivalence.py examples/CX09T1 \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26
python3 scripts/profile_scalar_f32_keys.py examples/CX09T1 \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26 --repeats 5 \
  --output cx09t1_f32_keys_d16.csv
```

## Active-state representation (v1.8)

Use `--active-state separate` for the byte-array reference path and `--active-state parent-sentinel` to encode inactivity as `parent[v] == u32::MAX`. See `ACTIVE_STATE_ABLATION.md` for the exact equivalence and profiling commands.

## v1.16: local H2 birth storage

```text
--local-h2-birth-state tagged|compact
```

`tagged` is the v1.16 reference default. `compact` stores local H2 births at
native scalar-key width and reserves the impossible +infinity ordered key for
the outside state. See `LOCAL_H2_BIRTH_STATE_INVESTIGATION.md`.

## v1.19 candidate: global H0 union-find layout

```text
--global-h0-uf-layout parent-rank|packed
```

`parent-rank` remains the reference/default in the candidate tree. `packed`
encodes root rank inside the `u32` parent word and removes the separate global
rank byte. Validate with
`validation/check_scalar_h0_global_uf_layout_equivalence.py` and profile with
`scripts/profile_scalar_h0_global_uf_layout.py`. See
`GLOBAL_H0_UF_LAYOUT_INVESTIGATION.md`.
