# v1.9 interface-state investigation

The scalar H0/H2 local persistence reducers historically maintain
`interface_rep: Vec<u32>` for every voxel in a slab. Only components that
reach the lower or upper z-face need such a representative, but the vector
costs four bytes per voxel regardless.

v1.9 adds:

```text
--interface-state vector
--interface-state root-invariant
```

`vector` is the reference implementation and remains the default.

In `root-invariant`, `interface_rep` is not allocated. Instead the local
union-find maintains the invariant:

> If a local component contains a slab-interface voxel, its union-find root is
> itself a slab-interface voxel.

The interface node corresponding to a root is then derived from the root's
local voxel index with `local_boundary_node_id`. When exactly one merging root
is an interface root, that root is forced to survive. When both or neither are
interface roots, normal union-by-rank chooses the survivor. For a forced-root
merge, the stored rank is raised to `max(rank_survivor, rank_child + 1)` so it
continues to bound tree height.

The persistence elder-rule state is independent of the physical union-find
root: the merged birth state is explicitly written to the selected survivor.
Interface merge/attach events are emitted before the physical root is chosen.
Changing the surviving interface representative is therefore expected to be
semantically neutral, but this is checked against the independent in-memory
scalar implementation rather than assumed.

## Correctness check

```bash
python3 validation/check_scalar_interface_state_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

This compares the in-memory oracle, `vector`, and `root-invariant` for both H0
and H2 as exact persistence multisets.

## Timing and RSS

```bash
python3 scripts/profile_scalar_interface_state.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_interface_state_d16.csv
```

H0 is benchmarked with the measured `--active-state separate`; H2 is
benchmarked with `--active-state parent-sentinel` so only interface state is
varied.

## Structural diagnostics

```bash
python3 scripts/profile_scalar_interface_state_diagnostics.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --output cx09t1_interface_state_diagnostics_d16.csv
```

Important fields are:

- `interface_state_bytes`: peak per-slab bytes used by the explicit interface
  representative vector (zero in root-invariant mode),
- `interface_forced_root_unions`: merges where an interface root had to survive
  over an interior root,
- `interface_interface_unions`: merges where both roots were interface roots,
- `max_rank_observed`: a compact signal for whether forced root choice worsens
  tree depth,
- `find_parent_steps` / `find_calls`: the measured path-traversal consequence.

Promotion requires exact output equality and a favorable wall/RSS tradeoff;
zero vector allocation alone is not sufficient.
