# v1.20 architectural optimization candidate

v1.20 is deliberately an **experimental architecture branch**. It does not change the v1.19 production defaults for H0 birth-buffer/event storage, hierarchical H0, or merge selection until CX09T1 equivalence and profiling have passed.

## Candidate changes

### H2 native F32 end-to-end

For an F32 stack with `--f32-key-mode native32`, H2 now retains the 4-byte `F32Key` through slab summaries, temporary runs, interface births, and compact global H2 reduction. The historical wide64 path remains available through `--f32-key-mode legacy64`. Native32 requires compact global H2 birth storage.

The global H2 packed layout remains an explicit ablation (`--global-h2-uf-layout packed`) until measured.

### H0 input-buffer reuse

`--h0-birth-buffer reuse-input` transfers the decoded F32 slab vector into union-find birth storage after the scalar order and boundary snapshots are constructed. The reference `copy` path is still the default.

### H0 direct event sinks

`--h0-event-storage direct` writes finalized local pairs, attach events, and interface events during the filtration sweep rather than buffering three whole-slab vectors. It currently requires native F32, `--h0-birth-buffer reuse-input`, and `--event-order verify`. The reference `buffered` path remains the default.

### Hierarchical exact H0

`h0-scalar-hierarchical` is an experimental in-memory exact H0 mode. Leaf slab summaries are composed with online binary fan-in. When two adjacent summaries are combined, their shared face becomes interior; only the outer faces survive as terminals. Components that cease touching an outer terminal are finalized during the composition.

The branch contains exact tests against unsplit/flat H0 and a CX09T1 validation script. An independent randomized model audit also passed 8,100 small 3-D cases spanning 6/26-connectivity, tied values, multiple slab depths, binary fan-in, and odd tails. This does not replace the Rust/CX09T1 gate; it separately checks the summary algebra. The runtime emits `max_live_summaries` and `max_pair_nodes` so the frontier bound is measured directly. The hierarchical preflight validates the pairwise frontier (at most four z-faces), not the cumulative flat interface-node count.

This prototype establishes the summary-composition invariant before a disk-backed hierarchical production implementation is attempted.

### Scale-aware merge selector

`--merge-strategy auto` selects `scan` for up to 32 event readers and `heap` above 32 readers. It emits `PROFILE_MERGE_SELECT`. The threshold is experimental and the production default remains `scan` until profile data establish a crossover.

## Required first gate

Build with Rust 1.88 and then run:

```bash
cargo fmt --all
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release --locked

scripts/check_v1_20_architecture_candidate.sh examples/CX09T1 target/release/betti_curves
```

The H0 hierarchical test uses native CX09T1 U16 data. The F32-specific H0/H2 tests stage a small exact-value F32 fixture from CX09T1; every U16 value is exactly representable in F32, so the filtration values are unchanged.

## Promotion policy

Do not promote any candidate solely from theoretical memory accounting. Promotion requires:

1. exact persistence equality with the independent in-memory implementation;
2. exact equality with the v1.19 reference streaming path where applicable;
3. configuration/storage profiles proving that the intended candidate path executed;
4. no material runtime regression on CX09T1 or the staged real-value F32 fixture;
5. measured peak-memory or temporary-I/O benefit for memory-oriented changes.

The hierarchical prototype has an additional requirement: exact equality across several slab depths, including odd/non-power-of-two fan-in cases in unit tests.
