# v1.4 local-sweep investigation

The CX09T1 v1.3 measurements showed that the local topology sweep accounts for roughly 80% or more of scalar-stream slab preparation. v1.4 therefore keeps the measured production settings (`scan + radix + verify + interior-fast`) and investigates the sweep itself.

## Representative active-check ablation

`NeighborhoodComponentPruner` constructs its representative mask only from neighbors for which `active[neighbor] != 0`. The subsequent active test in the persistence union helper is therefore logically redundant if the pruner invariant holds.

Two runtime modes are available:

- `--representative-active-check recheck` retains the old defensive load/test.
- `--representative-active-check trust-pruner` removes that repeated load/test.

The default remains `recheck` until the ablation is measured. `--neighbor-kernel interior-fast` is now the default because it was exactly persistence-equivalent and faster for both H0 and H2 on CX09T1.

## Diagnostic counters

`--sweep-diagnostics` enables operation counters. This mode is for diagnosis, not timing; the extra counters deliberately add overhead. The program emits a `PROFILE_SWEEP` line containing:

- pruning-mask calls;
- active-neighbor hits;
- representative visits;
- pruning-cache hits and misses;
- component-mask computations;
- union attempts, successful unions, and same-root attempts;
- union-find `find` calls and parent traversals;
- representative active rechecks and failures.

When diagnostics are enabled with `recheck`, any nonzero active-recheck failure aborts the run. This makes the invariant required by `trust-pruner` explicit on real data.

## Recommended CX09T1 experiment

First prove exact equivalence and the activity invariant:

```bash
python3 validation/check_scalar_representative_active_equivalence.py \
  examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

Then benchmark the recheck removal without diagnostics overhead:

```bash
python3 scripts/profile_scalar_representative_active.py \
  examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_representative_active_d16.csv
```

Finally obtain one operation-count snapshot:

```bash
python3 scripts/profile_scalar_sweep_diagnostics.py \
  examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --output cx09t1_sweep_diagnostics_d16.csv
```

The diagnostic CSV derives cache hit rate, active-neighbor and representative counts per voxel, representative retention, same-root union fraction, and average parent traversals per `find`. These ratios determine whether the next optimization should target pruning-cache behavior, union/find work, or active-state storage.


## CX09T1 v1.4 result

The representative-active invariant held with zero failures for H0 and H2, but removing the repeated active check was a performance wash. The default therefore remains `recheck`. The diagnostic counters showed a stronger target: H2 had about 65.9% same-root union attempts, and the conventional local union path performed two `find()` calls per representative edge. v1.5 therefore investigates carrying the current root across the representative-neighbor loop.
