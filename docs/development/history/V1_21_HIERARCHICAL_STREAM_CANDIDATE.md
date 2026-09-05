# v1.21 optimized disk-backed hierarchical H0 candidate

## Why this candidate exists

The v1.20 architecture gate passed exact CX09T1 equivalence, and the subsequent profile established three facts:

1. the H0 hierarchical summary algebra is exact and its measured pair frontier is exactly `4A`, independent of slab count;
2. native-F32 `reuse-input + direct` H0 leaves cut the staged-F32 peak RSS from about 26.24 MiB to 13.09 MiB (-50.1%) and improve median wall time by about 4.8%;
3. the v1.20 in-memory hierarchical prototype is intentionally slow/memory-heavy because it retains event vectors and uses the reference leaf kernel.

v1.21 therefore changes the implementation of the hierarchy, not the mathematics.

## Disk-backed summary

A live hierarchical summary stores only:

- the two exterior F32 face-value arrays;
- one monotone attach-event run;
- one monotone interface-merge run;
- its z-range and interface-node count.

Finalized persistence pairs are appended immediately to one common disk run and never retained by a parent summary.

When two adjacent summaries are combined, a temporary sparse cross-interface run is constructed. Five monotone streams are then scanned in exact priority order at each filtration value:

1. left attach;
2. right attach;
3. left interface;
4. right interface;
5. cross interface.

The shared faces become interior. Only the left lower face and right upper face survive into the parent summary. After the parent runs are flushed, both child event runs and the transient cross run are deleted.

## Pairwise union-find state

The pair reducer uses packed `u32` parent/rank words and native 4-byte F32 births. It does not allocate a terminal-representative vector. Instead, root selection preserves an exterior terminal as the root whenever a component touches one, so terminal identity can be inferred from the root ID itself.

The explicit pair state is therefore 8 bytes per child-interface node:

```text
packed parent/rank  4 bytes
F32 birth           4 bytes
---------------------------
                    8 bytes
```

For a full pair of ordinary two-face summaries, at most `4A` nodes are present. On the 3792 x 3792 target face this is 57,517,056 nodes, or about 0.429 GiB of explicit pair UF state. This replaces the flat d32 global-H0 state of about 13.71 GiB.

## Leaf producer

The experimental mode uses the already-validated native-F32 leaf path:

```text
--f32-key-mode native32
--h0-birth-buffer reuse-input
--h0-event-storage direct
--event-order verify
```

The input F32 slab buffer becomes component-birth storage after ordering and boundary snapshots are created, and local persistence actions are written directly to disk instead of accumulating three whole-slab event vectors.

## Mode

```text
h0-scalar-hierarchical-stream
```

This remains experimental. The flat production/reference implementation is unchanged.

## Required gate

```bash
cargo fmt --all
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release --locked

scripts/check_v1_21_hierarchical_stream.sh \
  examples/CX09T1 \
  target/release/betti_curves
```

The validator stages an exact-value F32 subset from CX09T1 and requires exact equality among in-memory H0, flat native32/direct streaming H0, and optimized hierarchical streaming H0. It additionally requires `max_pair_nodes <= 4A`, final interface state `=2A` for a multi-slice fixture, and 4-byte disk keys.

## Profiling gate

After exactness passes:

```bash
python3 scripts/profile_v1_21_hierarchical_stream.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slice-limit 0 \
  --slab-depths 8 16 32 \
  --repeats 3 \
  --output cx09t1_v1_21_hierarchical_stream.csv
```

The profile compares the optimized hierarchy with both the reference hierarchy and flat direct/reuse streaming on the same exact F32 rendering of CX09T1.
