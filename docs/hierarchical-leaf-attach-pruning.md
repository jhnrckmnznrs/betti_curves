# Hierarchical leaf attach pruning

## Motivation

A slab-local union between one boundary-visible component and one component that does not touch either z-boundary has two different outcomes.

If the internal branch becomes the elder branch of the merged component, the branch state visible at the slab boundary changes. That transition must remain an `Attach` event because a higher fan-in level needs to see it.

If the boundary-visible branch is already elder, the internal branch dies at this slab-local union and can never become boundary-visible later. The old leaf code nevertheless exported this case as an `Attach`, forcing higher fan-in levels to replay it until some pair reducer rediscovered that the branch was non-promoting.

The v10 candidate removes that redundant replay from the hierarchical path.

## Invariant

Let `c` be an internal branch and `p` the branch currently visible on a slab boundary component. Suppose the local union occurs at filtration value `d` and the elder rule chooses `p` over `c`.

Then:

1. `c` has no slab-boundary representative before the union.
2. After the union, `c` is not the elder state of the boundary-visible component.
3. Therefore no later hierarchical summary can expose `c` as a live boundary branch.
4. The death value `d` of `c` is final at the leaf.
5. A higher-level connection can only change the *parent identity* of this finalized death by replacing provisional parent `p` earlier in filtration order.

The hierarchical root path already performs exactly this deferred parent repair. H0 redirects through strictly smaller death thresholds; H2 redirects through strictly larger thresholds. Equal-threshold interface/cross connections occur after attach processing and therefore do not alter this local attach's positive-persistence parent at the same filtration value.

Consequently the hierarchical leaf may store `c -> p` as finalized history immediately instead of exporting an `Attach(c)` event.

## Scope

The optimization is deliberately restricted to hierarchical branch-tree computation. The flat/in-memory slab summary generator keeps the original unconditional attach behavior and remains the independent equivalence reference.

The hierarchical implementation retains an A/B switch:

```bash
BETTI_HIER_EARLY_FINALIZE_LEAF_ATTACHES=0
```

With the variable unset, early finalization is enabled.

## Profiling counters

Hierarchy profile lines report:

- `leaf_one_boundary_internal`: slab-local unions with exactly one z-boundary-visible component;
- `leaf_finalized_attach_early`: those unions converted directly to finalized history;
- `leaf_propagated_attach`: those that genuinely changed the elder boundary state and remained attaches.

Per-combine lines additionally report `input_attach` and `input_interface`, which makes the reduction in fan-in workload directly measurable.

## Correctness gates

Run the symbolic repair check:

```bash
python3 validation/check_leaf_attach_early_finalization_model.py --cases 50000
```

Then require exact flat-vs-hierarchical equivalence on the real validation volume for both H0 and H2 before profiling.

## Interaction with inline history contraction

Early attach finalization and history contraction are complementary. v10
removed redundant `Attach` replay but still materialized those deaths in the
leaf's `local_merge_events` vector before the central v8 contractor discarded
most diagonal events. The v11 candidate feeds all leaf-local finalized deaths
through an inline contractor instead. In particular, a non-promoting attach
that is also zero-persistence can now be finalized, parent-repaired if needed,
and discarded without ever entering the central finalized-event buckets.

Use `BETTI_HIER_INLINE_LEAF_HISTORY=0` to retain the v10 storage behavior while
keeping v10 early attach finalization enabled.
