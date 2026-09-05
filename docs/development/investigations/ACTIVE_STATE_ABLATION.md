# v1.8 active-state ablation

This revision tests whether the slab-local `Vec<u8>` active array can be removed from exact scalar H0/H2 persistence.

Two representations are available:

- `--active-state separate` (reference/default): the historical byte-per-voxel active array.
- `--active-state parent-sentinel`: no active byte array is allocated. An inactive UF entry has `parent[v] == u32::MAX`; activation sets `parent[v] = v`.

The pruning scan now first returns an owned list of at most 26 representative neighbor indices, then the persistence kernel performs unions. This avoids unsafe aliasing when parent state is simultaneously the activity source and the mutable union-find state. Both active-state modes use the same representative-list API so the A/B comparison isolates the state representation.

## Correctness

```bash
python3 validation/check_scalar_active_state_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26
```

This compares in-memory exact scalar persistence with both streaming active-state representations for H0 and H2.

## Timing and peak RSS

```bash
python3 scripts/profile_scalar_active_state.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26 \
  --repeats 5 --output cx09t1_active_state_d16.csv
```

## Operation counters

```bash
python3 scripts/profile_scalar_active_state_diagnostics.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26 \
  --output cx09t1_active_state_diagnostics_d16.csv
```

The diagnostic path requires both representations to observe the same number of activity checks, active-neighbor hits, and persistence intervals.
