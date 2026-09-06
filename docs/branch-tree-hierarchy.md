# Hierarchical branch trees

The legacy `branch-tree-h0` and `branch-tree-h2` reducers retain all slab-interface state until the final global sweep. The hierarchical modes reduce adjacent slab summaries online:

```text
branch-tree-h0-hierarchical
branch-tree-h2-hierarchical
```

The current branch-tree implementation follows the existing integer branch-tree path and therefore accepts the U16-compatible TIFF reader; native scalar/F32 branch identities are a separate follow-up after this structural equivalence gate.

They use the same binary-counter fan-in pattern as hierarchical persistence. A completed block keeps only its two external z-faces plus the sparse events that can still alter the elder branch seen at those faces. Internal branch deaths are finalized immediately.

## Why plateau-canonical parenting is required

The historical branch tree is deterministic only after fixing an order for equal-valued events. If three positive branches enter the same merge plateau at threshold `t`, different spanning-tree edge orders can produce either a chain or a star even though the filtration and persistence barcode are identical.

The hierarchical modes remove that representation artifact. After reduction, if a branch and its parent die at the same H0 threshold, the child is reparented to the first ancestor that survives past that threshold. For H2 the analogous rule uses the exported branch-birth threshold. The result is a **plateau-canonical elder-rule branch tree**.

This makes the tree invariant to the particular equal-threshold spanning forest used at an interface. Therefore the existing filtration-preserving cross-interface sparsifier remains valid for the hierarchical tree: it preserves the connected-component partition of every threshold prefix, which is exactly the information needed to identify a merge plateau and its oldest survivor.

## H0 summary rule

For a sublevel H0 component, the older branch is the lexicographically smaller `(birth, branch_id)` pair.

When an internal branch reaches a component represented on the parent block boundary at merge value `d`:

- if the boundary branch is already older, the internal branch death and parent are final;
- if the internal branch is older, only the boundary-state transition propagates because it changes the elder branch visible on the parent boundary from `d` onward; the boundary branch death is **not** finalized at that level;
- if both components touch the parent boundary, their interface union propagates.

The second case is intentionally split into two effects. The elder label must change immediately inside the pair reducer, but the death of the branch that was previously visible on the boundary remains provisional: a higher-level connection can still reach that component at a smaller filtration value. The death becomes final only when that component loses its parent-boundary representative, or when the root summary is reduced in the full-volume context. Replaying the same propagated transition through several fan-in levels therefore cannot kill the same branch repeatedly.

This is the branch-tree analogue of elder-dominated attachment pruning.

## H2 summary rule

H2 is represented through the complementary superlevel background problem. The distinguished outside component is older than every finite branch; among finite branches, larger background birth is older (branch ID breaks ties).

The same propagation rule applies, with an additional outside state:

- finite branches dominated by an older terminal branch can be finalized;
- a newly older finite branch must propagate without prematurely finalizing the finite branch that was still visible on the parent boundary;
- an outside connection touching the parent boundary must likewise propagate as a state change, with the boundary-visible finite death deferred until it is globally determined;
- outside-only internal components can be discarded after their finite child branch has been finalized.

Equal-value priority is `outside -> finite attach -> interface -> cross`.

## Memory bound, root-event storage, and bounded leaf parallelism

Let `A = width * height`. Each child summary has at most two z-faces, so one pairwise reconciliation uses at most approximately `4A` interface nodes. Only `O(log(number_of_slabs))` completed summaries are live in the online binary fan-in.

Finalized interior branch deaths are bucketed by their `u16` filtration value as soon as they become final. The hierarchical root reducer consumes those buckets directly. It therefore does **not** first retain one flat vector of all finalized deaths and then allocate a second threshold-bucketed copy during global reduction. This removes the previous duplicate `O(number_of_finalized_branches)` event representation at the root while preserving the exact same within-threshold event order.

Leaf slabs are processed in deterministic bounded parallel batches. The default worker count is at most four Rayon workers and is reduced when a conservative per-leaf memory estimate would exceed a 512 MiB concurrent-leaf budget. The completed leaf summaries are reordered by slab ID before entering the serial binary fan-in, so parallel execution cannot change branch-tree semantics. Two environment variables are available for controlled benchmarking or memory-constrained machines:

```bash
BETTI_HIER_LEAF_WORKERS=1   # explicit worker count, capped by Rayon threads
BETTI_HIER_LEAF_BUDGET_MB=256
```

`BETTI_HIER_LEAF_WORKERS=1` reproduces serial leaf scheduling and is useful as a correctness/performance baseline. For very large slabs the automatic memory cap may also choose one worker.

The current implementation still returns the legacy in-memory `MergeTree`, so final node/edge materialization remains `O(number_of_retained_branches)`. A future stable-ID streaming writer can remove that output-materialization term as well.

## Validation

Build the release binary and compare the hierarchical result with the existing flat reducer after canonicalizing equal-threshold plateaus:

```bash
python3 validation/check_branch_tree_hierarchical_equivalence.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 8 \
  --foreground-connectivity 26
```

The validator checks the complete node table: retained node IDs, parent IDs, birth values, and death values for both H0 and H2.

## Deferred parent resolution

An interior branch death can be finalized before the eventual fate of its provisional parent is known.
The hierarchical reducer therefore does not assume that a locally named parent is still alive when the
final tree is materialized. It watches only branch IDs referenced by finalized interior merges and, at
each later filtration level, advances those references through already-observed deaths. H0 performs this
in ascending sublevel order; H2 performs the dual operation in descending superlevel order. Same-level
relations are intentionally left to plateau canonicalization. This keeps ancestry exact without retaining
all eliminated interface branches.

## Streaming ordered fan-in

Adjacent summaries are reconciled without first constructing one merged event vector. A cursor merges the already filtration-ordered attach, interface, and cross-interface streams and feeds each selected event immediately to the pair union-find. The cursor uses the same tie order as the previous stable-sort/linear-merge path, so this changes storage and memory traffic rather than branch-tree semantics. Set `BETTI_HIER_MATERIALIZE_FANIN=1` to materialize the cursor output for reference A/B runs.

## Finalized-history contraction

The optimized hierarchical branch-tree path can contract finalized history
before root replay. The filtration-time invariant, proof boundary, and exact
H0/H2 redirect inequalities are documented in
[`hierarchical-history-contraction.md`](hierarchical-history-contraction.md).
