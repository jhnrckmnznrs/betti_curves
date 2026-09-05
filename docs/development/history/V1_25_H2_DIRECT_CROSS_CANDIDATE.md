# v1.25 H2 direct cross-interface candidate

v1.25 isolates the next H2 hierarchy optimization after v1.24: remove the
write/flush/reopen/read cycle for sparse cross-interface forests.

## Motivation

The v1.24 CX09T1 profile showed that hierarchical H2 is already exact, faster,
and lower-memory than flat native32/packed H2. Attach propagation is >99.7%
pruned, while the pairwise UF is only 8 bytes per interface node and bounded by
4 face areas. Cross-interface forests are therefore a low-risk remaining I/O
path: each fan-in writes a sorted sparse forest to disk only to reopen and
consume it immediately.

## New ablation

```
--h2-hier-cross-storage disk
--h2-hier-cross-storage direct
```

`disk` is the v1.24 reference. `direct` consumes the cross forest directly from
the interface sparsifier callback.

For a cross edge with filtration value `v`, direct mode first drains all child
summary events with values greater than `v`, then all child outside/attach/
interface events at exactly `v`, and only then applies the cross edge. Thus the
existing deterministic H2 priority

```
outside -> attach -> interface -> cross
```

is preserved exactly. Cross edges themselves remain in the sparsifier's
existing deterministic descending order.

Direct mode does not add a persistent event vector and does not change the
pairwise 4A / 8-byte-per-node memory bound.

## Validation

Build and run:

```
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release --locked

validation/check_v1_25_h2_cross_direct.py \
  examples/CX09T1 \
  --binary target/release/betti_curves
```

The gate compares the exact persistence multiset from hierarchical `disk` and
`direct` against an independent in-memory H2 oracle at d4/d8/d16.

Then profile:

```
python3 scripts/profile_v1_25_h2_cross_storage.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slice-limit 0 \
  --slab-depths 8 16 32 \
  --repeats 3 \
  --output cx09t1_v1_25_h2_cross_storage.csv
```

Promotion requires exact canonical persistence, no RSS regression, and a
repeatable reduction in wall/system time or filesystem I/O. If direct is
neutral on CX09T1 but removes substantial cross-run traffic, it remains worth a
real 3792x3792 prefix test because cross forests scale with face area.
