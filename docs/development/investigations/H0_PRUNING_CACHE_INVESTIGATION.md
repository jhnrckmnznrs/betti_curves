# H0 pruning-cache investigation (v1.7)

The 26-neighbor local-pruning kernel maps each active-neighbor mask to one
representative per connected component of the induced neighborhood. Historically
this representative mask was stored in a fixed 65,536-entry direct-mapped cache.
CX09T1 v1.4 diagnostics measured only about a 15.9% hit rate, so v1.7 makes the
cache size an H0-specific ablation instead of assuming the historical choice is
optimal.

## Runtime control

For scalar H0 streaming only:

```text
--h0-pruning-cache off|4k|16k|64k|256k
```

The enabled sizes contain 4,096, 16,384, 65,536, or 262,144 key/value entries.
Each entry uses one `u32` key and one `u32` representative mask, corresponding to
approximately 32 KiB, 128 KiB, 512 KiB, or 2 MiB per active H0 pruner. `off`
allocates no cache and performs no cache lookup.

The historical 64K cache remains the default until the ablation is measured.
For 6-connectivity this option has no effect because the neighborhood mask is
already minimal and no pattern cache is used.

## Diagnostics

With `--sweep-diagnostics`, H0 now emits:

```text
PROFILE_CACHE scalar_h0_stream mode=... entries=... storage_bytes=...
```

and the sweep diagnostics include `pruning_cache_lookups`. For an enabled cache,

```text
lookups = hits + misses
component_mask_computations = misses
```

For `off`, lookups, hits, and misses are all zero; every nonempty 26-neighbor
mask is computed directly.

## Correctness and measurement

Use:

```bash
python3 validation/check_scalar_h0_pruning_cache_equivalence.py STACK \
  --binary target/release/betti_curves --slab-depth 16 --foreground-connectivity 26
```

Then benchmark without diagnostics overhead:

```bash
python3 scripts/profile_scalar_h0_pruning_cache.py STACK \
  --binary target/release/betti_curves --slab-depth 16 --foreground-connectivity 26 \
  --repeats 5 --output cx09t1_h0_pruning_cache_d16.csv
```

And collect operation counts separately:

```bash
python3 scripts/profile_scalar_h0_pruning_cache_diagnostics.py STACK \
  --binary target/release/betti_curves --slab-depth 16 --foreground-connectivity 26 \
  --output cx09t1_h0_pruning_cache_diagnostics_d16.csv
```

The timing profiler refuses success unless every cache size produces the same
canonical H0 persistence multiset.
