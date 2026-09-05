# Scalar-stream preparation profiling (v1.3)

This revision adds phase-level instrumentation for exact scalar H0/H2 persistence streams and an ablation of the local-neighborhood kernel.

## Production/reference settings

The v1.3 experiment established `interior-fast` as the measured production kernel on CX09T1:

```text
--merge-strategy scan
--interface-order radix
--event-order verify
--f32-key-mode native32
--neighbor-kernel interior-fast
```

`generic` remains available as the reference ablation path.

## Detailed preparation phases

Each scalar persistence stream now emits one `PROFILE_PREP` line containing:

- `decode_seconds`: time inside TIFF `read_image()` calls.
- `key_conversion_seconds`: conversion from decoded scalar samples to ordered keys.
- `slab_copy_seconds`: extra per-slice copy into the legacy-wide slab buffer. Native F32 appends directly and normally reports zero here.
- `scalar_order_seconds`: counting/radix ordering of slab voxels.
- `local_sweep_seconds`: local activation sweep, neighborhood pruning, union-find persistence bookkeeping, and boundary-face extraction.
- `local_run_write_seconds`: local finite-pair/interface-birth/event-run writing and final flushes.
- `cross_interface_seconds`: cross-slab interface sparsification together with its buffered edge-run output.
- `unaccounted_seconds`: preparation wall time not covered by the explicit timers, including setup/metadata/assertion/allocation overhead outside the timed regions.
- `total_voxels` and `interior_fast_voxels`: coverage counters for the neighbor-kernel experiment.

The phase timers are intentionally outside the per-voxel hot loop.

## Interior-neighbor fast path

The new option is:

```text
--neighbor-kernel <generic|interior-fast>
```

For `interior-fast`, a voxel uses precomputed linear neighbor offsets only when it is strictly inside all six **local slab** faces. Boundary voxels use the original coordinate-checked method. The active-neighborhood bit order and representative-mask logic are unchanged.

The fast path therefore changes only how valid neighbor indices are formed; persistence ordering and union-find rules remain identical.

## Correctness check

On a representative stack small enough for the in-memory scalar implementation:

```bash
python3 validation/check_scalar_neighbor_kernel_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

This requires exact multiset agreement among:

1. in-memory scalar persistence,
2. streaming `generic`, and
3. streaming `interior-fast`.

## Balanced profiler

```bash
python3 scripts/profile_scalar_neighbor_kernel.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_neighbor_kernel_d16.csv
```

The profiler warms both paths, alternates execution order, records `/usr/bin/time -v` metrics and all internal preparation phases, hashes canonical persistence multisets, and fails if the two kernels disagree.

On CX09T1 at slab depth 16, `interior-fast` reduced the H0 local sweep by about 15% and H2 by about 7%, with exact persistence agreement. It is therefore the default in later revisions; `generic` remains available for regression testing.
