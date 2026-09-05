# Production consolidation status

The v1.13 production release freezes the measured defaults before another
optimization cycle. H0 uses scan/radix/verify, native32 F32 keys, the
interior-fast neighborhood kernel, root-carrying unions, a 64K pruning cache,
separate active state, root-invariant interface state, and packed local UF. H2
uses the same common choices plus the direct-parent root shortcut,
parent-sentinel active state, packed local UF, and a single pre-reduction
`malloc_trim(0)` on Linux/glibc. Historical sections below describe the
experiments as they were introduced and should not be read as current defaults.

# Scalar-stream optimization audit: v1.2 native-F32 revision

This revision turns the first scalar H0/H2 optimization batch into a controlled
experiment. The mathematical filtration, connectivity convention, elder rule,
and persistence output format are unchanged.

## Runtime ablation controls

The scalar persistence stream modes now accept:

```text
--merge-strategy <scan|heap>
--interface-order <comparison|radix>
--event-order <resort|verify>
```

The CX09T1 factorial ablation selected `scan + radix + verify`, which is now
the production algorithmic default.
The `scan + comparison + resort` combination reconstructs the original choices
for the three changed code paths, allowing same-binary comparisons without a
separately preserved baseline executable. H2 `resort` sorts the slab vectors
in place, matching its original implementation rather than introducing an
extra diagnostic copy.

## Correctness controls

- Comparison and radix interface orders have Rust unit-test oracles, including
  tied scalar values and superlevel reversal.
- `verify` checks the local event monotonicity invariant while serializing.
- `resort` deliberately restores the old sorting step and therefore provides an
  independent route to the same run ordering.
- `validation/check_scalar_stream_equivalence.py` can run any selected ablation
  combination against the in-memory exact scalar H0/H2 implementation.
- `validation/check_scalar_stream_ablation_equivalence.py` checks all eight
  combinations against the in-memory oracle in one command (or the legacy/v1
  endpoints with `--quick`).
- `scripts/profile_scalar_stream_ablation.py` canonicalizes every persistence
  CSV as a multiset and fails unless all measured configurations agree exactly
  within each homology degree.

## Profiling controls

`scripts/profile_scalar_stream.py` now performs warm-up runs by default and
alternates baseline/candidate order between repeats. The new
`scripts/profile_scalar_stream_ablation.py` performs the full 2 x 2 x 2
factorial experiment on one binary, rotates/reverses execution order, captures
`/usr/bin/time -v` counters, parses internal phase timings, and reports all
eight cells separately.

Recommended representative experiment:

```bash
cargo test --all-targets
cargo build --release --locked

python3 validation/check_scalar_stream_equivalence.py examples/CX09T1 \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26

python3 scripts/profile_scalar_stream_ablation.py examples/CX09T1 \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26 --repeats 3 \
  --output cx09t1_ablation_d16.csv
```

## Deliberately deferred

Native-width F32 storage is implemented behind `--f32-key-mode native32`.
v1.18 extends the H0 candidate end-to-end through boundary summaries, disk
runs, cross-interface events, and the global H0 birth vector. The legacy-wide
path remains available as `legacy64` for same-binary correctness and performance
comparisons. H2 still widens at its existing global/disk boundary and is not
changed by the v1.18 H0 experiment.

Direct-to-run local event sinks, active-state compression, interface-root
invariants, interior-neighbor kernels, parallel radix ordering, bounded-fan-in
merges, and automatic slab-depth selection remain deferred.

## Validation limitation in the preparation environment

The Python validation/profiling scripts can be syntax-tested here, but the
preparation environment has no Rust toolchain. A local `cargo test --all-targets`
and release build are therefore required before performance conclusions are
accepted.


## v1.3 measurement target

The native-F32 experiment was effectively tied with the legacy-wide path on CX09T1, so v1.3 does not further optimize scalar key width. Instead it decomposes slab preparation into measured phases and introduces an opt-in interior-neighbor kernel. The generic kernel remains the reference/default until the factorial correctness check and balanced profiler establish a real wall-clock gain.

The interior kernel is deliberately narrow: only voxels strictly inside all six local slab faces use precomputed linear offsets; boundary voxels retain the previous coordinate-and-bounds implementation. This preserves a straightforward correctness oracle and avoids unsafe code.

### v1.7: H0 neighborhood-pruning cache

The cache is now an explicit H0 ablation (`off`, 4K, 16K, 64K, 256K entries).
This does not alter the representative-mask computation or persistence logic;
a collision or disabled cache only causes recomputation. The 64K historical
configuration remains the default until measured. The v1.6 H2 parent shortcut
is promoted to the default after exact-equivalence and profiling on CX09T1.

## v1.8: active-state representation

The next measured target is the dedicated `Vec<u8>` activity state used during the local scalar persistence sweep. v1.8 adds a parent-sentinel alternative but does not promote it by default. The experiment is accepted only if H0/H2 persistence matches the independent in-memory oracle and real-data wall/RSS measurements improve without increasing UF work.
