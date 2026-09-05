# Performance profiling

## Current performance model

The large-volume engine has four recurring cost classes:

1. TIFF decode and scalar ordering;
2. local neighbor/union-find sweep;
3. interface sparsification and hierarchical summary reduction;
4. temporary-run filesystem traffic.

Recent optimization work moved H0/H2 away from slab-count-dependent global union-find state. On the real H0 prefix test, RAM remained nearly constant when z-depth doubled, while earlier unpruned hierarchy versions showed I/O amplification. Elder-dominated attach finalization removed most of that replay traffic.

## Flamegraphs

Install `cargo-flamegraph` and Linux perf tooling, then run:

```bash
cargo install flamegraph
scripts/profile_flamegraph.sh /path/to/slices 16 26 h2-scalar-hierarchical-stream
```

The helper builds a release binary, records the exact command and git commit, and writes the SVG plus metadata under `profiles/`.

Do not commit a flamegraph generated from an arbitrary laptop run as a universal performance claim. For published flamegraphs, record CPU model, kernel, Rust version, input shape/type, thread count, slab depth, and scratch filesystem.

## Memory profiling

Use the benchmark suite for comparable peak-RSS and filesystem counters:

```bash
/usr/bin/time -v target/release/betti_curves ...
```

For allocator-specific investigation on Linux/glibc, the code also contains targeted H2 phase-trim diagnostics. Those are development tools, not correctness requirements.

## What to look for

- high `system` time or filesystem counters: temporary-run I/O / page-cache pressure;
- RSS growth with z-depth in a hierarchical mode: likely violation of the bounded-summary invariant;
- RSS growth with thread count: multiple slabs resident concurrently;
- large prepare time: scalar order / TIFF decode / local sweep;
- large reduce time: interface fan-in, event replay, or root-state overhead.

## Measured profiling observations

The committed development profiles show two examples of optimization driven by system-level evidence rather than microbenchmarks alone:

- H0 elder-dominated attach pruning reduced propagated attach records on CX09T1 by more than 99.9%, cutting system time and eliminating most repeated summary I/O. On the 3792×3792 real prefix, peak scratch stayed near 4 GiB when z-depth doubled from 64 to 128 slices.
- H2 direct-cross consumption reduced filesystem outputs by roughly 17–19% on CX09T1 while preserving the exact 220,093-interval barcode. Outside-dominated structural pruning then reduced hierarchical summary storage by about one third at d8/d16.

These observations motivate the current architecture but should not be treated as portable CPU benchmarks. Re-run the public suite on the target machine before making hardware-specific claims.


## Hierarchical branch-tree stage timers

The hierarchical H0/H2 modes emit `PROFILE_BRANCH_H0_HIERARCHY` / `PROFILE_BRANCH_H2_HIERARCHY` lines with wall-clock stage counters. The main fields are:

- `leaf_batch_seconds`: wall time spent running leaf-slab batches, including TIFF reads and leaf computation.
- `leaf_read_thread_seconds`: sum of per-leaf read times across worker threads; this can exceed wall time under parallel execution.
- `leaf_compute_thread_seconds`: sum of per-leaf topology-computation times across worker threads; this can also exceed wall time.
- `bucket_seconds`: time moving finalized local events into compact threshold buckets.
- `fan_in_seconds`: time spent in binary-carry fan-in combines triggered while ingesting leaves.
- `root_combine_seconds`: time combining the residual summaries after the carry slots are drained.
- `root_reduce_seconds`: time for the final global interface union-find/deferred-parent reduction.
- `canonical_seconds`: time for plateau-canonical tree normalization.

Each `PROFILE_BRANCH_*_HIER_COMBINE` line also reports `setup_seconds`, `cross_seconds`, `sort_seconds`, `reduce_seconds`, `finish_seconds`, and `total_seconds`. These are intended for bottleneck attribution only; run multiple repetitions before treating small differences as meaningful.
