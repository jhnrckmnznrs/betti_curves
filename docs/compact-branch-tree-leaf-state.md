# Compact branch-tree leaf state

This experimental optimization targets the leaf sweep rather than widening the upper hierarchy.

## Motivation

For an integer H0/H2 branch-tree leaf, the previous hot state contained a `u32` parent, a separate `u8` rank, a full branch object, a `u32` interface representative, and a separate active byte per voxel. The branch object is redundant: every finite branch is born at a voxel in the current slab, so its global ID and birth value are recoverable from one local voxel ID.

The compact layout stores:

- packed parent/rank state with the inactive state encoded in the parent word;
- one `u32` elder local voxel ID per root (`u32::MAX` is the H2 Outside sentinel);
- one `u32` interface representative.

The hot UF/activation metadata therefore falls from roughly 26 bytes/voxel to 12 bytes/voxel before allocator overhead and the filtration-order arrays are counted.

## Exactness invariant

For H0, let `e` be the stored elder local voxel index. Its branch is reconstructed as

`(birth[e], global_base + e)`.

Comparing two local elders by `(birth[e], e)` is identical to comparing their full `(birth, global_id)` branch keys because `global_base` is common to every voxel in the slab.

For H2, `u32::MAX` represents Outside. Finite elders compare by descending birth, then ascending local ID, which is again identical to the full global-ID rule inside one slab.

No exported event format changes.

## Neighbor kernel

The leaf sweep now uses `NeighborhoodComponentPruner::representative_neighbors_interior_by` for voxels strictly inside the slab. It precomputes linear neighbor offsets and avoids repeated coordinate-bound checks in the dominant interior case. Boundary voxels continue to use the checked generic path.

The current UF root is carried across the representative-neighbor unions for one newly activated voxel, avoiding a redundant `find` of that voxel for each representative.

## Why this helps multi-slab fusion

A wider upper-level combine still has to replay leaf interface trees. A deeper leaf eliminates those intermediate interfaces at their source. If compact state permits the same worker count at d16 or d32, then doubling slab depth approximately halves the number of leaf summaries and physical hierarchy seams; quadrupling it quarters them.

This is the preferred route to "fused multi-slab assimilation" because it preserves one globally ordered leaf filtration instead of trying to finish one partial summary and later insert events that belong at earlier thresholds.

## Validation

`validation/check_compact_leaf_state_model.py` compares the compact elder representation against the legacy full-branch semantics under randomized H0/H2 unions, ties, boundary visibility, early-finalization policy, and H2 Outside states.

Real-volume acceptance still requires exact comparison against the frozen v12 hierarchical checkpoint for H0 and H2 at each candidate slab depth.
