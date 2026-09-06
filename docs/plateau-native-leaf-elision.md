# Plateau-native leaf zero-death elision

The hierarchical branch-tree leaf sweep is deterministic inside each `u16`
threshold. Historically, every union that killed a finite branch produced a
leaf-history record and the inline history contractor then removed records with
zero persistence (`birth == death`). On CX09T1 at slab depth 16 this meant
roughly 29 million leaf-history inputs per dimension even though fewer than
200,000 survived a leaf.

## Why whole-plateau preactivation is not used

The exact `NeighborhoodComponentPruner` assumes that one voxel is activated at
a time and that every edge among already-active voxels has already been
processed. Pre-activating an entire equal-value plateau would violate that
invariant and could allow local-neighborhood pruning to omit an edge whose
alternative path still contains an unprocessed plateau-to-older edge.

v16 therefore keeps the sequential activation order and existing neighborhood
pruner unchanged.

## Elision invariant

Let `t` be the current H0 filtration value (processed in nondecreasing order),
or the current H2 value (processed in nonincreasing order). Suppose a local
union kills a finite branch `c` with `birth(c) = t`.

Such a branch cannot already be the provisional parent of an earlier retained
positive-persistence leaf event:

- before `t`, `c` has not been born and therefore cannot be an active parent;
- at `t`, a positive-persistence child must have a strictly older birth than
  `t`, so a branch born at `t` cannot be its elder parent;
- equal-threshold children born at `t` have zero persistence and are themselves
  contracted.

Therefore the local `[t,t)` death has no retained positive parent references to
repair. It can be counted as contracted and discarded directly at the union
decision, before constructing a `LocalH0Merge`/`LocalH2Merge` or entering the
leaf history contractor.

Boundary semantics are not elided. Same-threshold unions that emit interface
merges or a boundary-state-changing attach continue to emit those transitions.
H2 Outside transitions are also unchanged.

## A/B switch

The optimization is enabled by default on the hierarchical inline-history path.
Set

```bash
BETTI_HIER_PLATEAU_NATIVE_LEAF=0
```

to restore the v15 local-history behavior in the same binary.

## Validation

`validation/check_plateau_native_leaf_elision_model.py` generates monotone
elder-rule H0/H2 merge streams (including H2 Outside merges) and verifies that
pre-history zero-death elision yields exactly the same retained positive events
and repair count as the generic v15 contractor.

The end-to-end correctness gate remains exact hierarchical tree equality against
the frozen validated checkpoint.
