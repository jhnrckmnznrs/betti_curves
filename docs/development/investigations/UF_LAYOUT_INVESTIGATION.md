# v1.10 local union-find layout investigation

This revision isolates the local scalar persistence union-find representation.

## Strategies

- `--uf-layout parent-rank` keeps the historical `Vec<u32>` parent array plus a
  `Vec<u8>` rank array.
- `--uf-layout packed` stores either a non-root parent index or tagged root-rank
  metadata in the same `u32` word. The rank vector is not allocated.

The packed representation reserves the high bit as the root tag. Consequently
it supports at most 2^31 voxels per local slab. The reference parent-rank
layout retains the previous u32 slab-index limit.

H0 benchmarking uses the measured best supporting configuration: separate
active-state bytes and root-invariant interface state. H2 benchmarking uses
parent-sentinel active state and root-invariant interface state. Both use the
previously selected scan/radix/verify, interior-fast and root-carrying paths;
H2 also uses the parent shortcut.

## Correctness

Run:

```bash
python3 validation/check_scalar_uf_layout_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves --slab-depth 16 --foreground-connectivity 26
```

This compares parent-rank and packed stream persistence against the independent
in-memory scalar persistence result for H0 and H2.

## Timing and RSS

```bash
python3 scripts/profile_scalar_uf_layout.py examples/CX09T1/ \
  --binary target/release/betti_curves --slab-depth 16 --foreground-connectivity 26 \
  --repeats 5 --output cx09t1_uf_layout_d16.csv
```

## Structural diagnostics

```bash
python3 scripts/profile_scalar_uf_layout_diagnostics.py examples/CX09T1/ \
  --binary target/release/betti_curves --slab-depth 16 --foreground-connectivity 26 \
  --output cx09t1_uf_layout_diagnostics_d16.csv
```

The diagnostic requires the packed path to report zero rank-state bytes and
checks that union-attempt/success/same-root counts remain unchanged. It records
`find_calls`, `find_parent_steps`, `max_rank_observed`, parent-state bytes and
rank-state bytes for both layouts.
