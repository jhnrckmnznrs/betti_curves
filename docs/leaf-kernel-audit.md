# Leaf-kernel audit

The hierarchical branch-tree leaf reducer has an opt-in diagnostic mode for
locating the next structural bottleneck after plateau-native zero-history
elision.

Enable it with:

```bash
BETTI_HIER_LEAF_KERNEL_AUDIT=1
```

The audit does **not** change filtration order, neighborhood pruning, union-find
semantics, branch ancestry, or exported summaries. Exact counters are collected
for every leaf operation. Timing is sampled rather than measured around every
voxel so that the diagnostic itself does not dominate the kernel.

The default timing stride is 4096 activated voxels. Override it with:

```bash
BETTI_HIER_LEAF_AUDIT_SAMPLE_STRIDE=8192
```

## Exact counters

The `PROFILE_BRANCH_H0_LEAF_KERNEL_AUDIT` and
`PROFILE_BRANCH_H2_LEAF_KERNEL_AUDIT` lines report:

- activated voxels and interior/local-boundary voxel counts;
- interface activations and, for H2, global-boundary activations;
- active-state checks performed by the exact neighborhood pruner;
- active-neighbor hits before local connectivity pruning;
- representative neighbors retained after pruning;
- 26-neighborhood pruning-cache lookups, hits, misses, and mask computations;
- union attempts, successful unions, and already-connected no-op attempts;
- successful unions classified as local/local, boundary/internal, or
  interface/interface;
- H2 unions whose elder state includes Outside;
- reuse versus re-find of the carried current root;
- union-find `find` calls, total traversed parent edges, and maximum observed
  parent depth.

These counters answer three questions directly:

1. **Is neighborhood pruning still the main opportunity?** Compare active-neighbor
   hits with representative neighbors.
2. **Are many retained representative edges globally redundant?** Compare union
   attempts with successful unions.
3. **Is union-find itself expensive?** Inspect mean/max find depth and how often
   the carried current root needs to be found again.

## Sampled timing

For every `BETTI_HIER_LEAF_AUDIT_SAMPLE_STRIDE`-th activation, the reducer times
three contiguous regions:

1. activation plus interface/global-boundary bookkeeping;
2. active-neighborhood construction and representative pruning;
3. union-find plus branch/interface action handling.

The external profiler converts these totals into sampled nanoseconds per voxel.
They are diagnostic proportions, not replacement end-to-end benchmarks. Use
normal audit-disabled runs for production timing.

Bucket construction is timed once per leaf and reported exactly as the sum of
per-leaf durations.

## Correctness gate

Because v17 is instrumentation-only, the strongest check is same-binary output
with audit enabled versus disabled:

```bash
BETTI_HIER_LEAF_KERNEL_AUDIT=1 \
python3 validation/check_branch_tree_against_validated_hierarchical.py \
  examples/CX09T1 \
  --candidate-binary target/release/betti_curves \
  --reference-binary target/release/betti_curves \
  --reference-env BETTI_HIER_LEAF_KERNEL_AUDIT=0 \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --dimensions h0 h2
```

For performance diagnosis, use H0 at the current fast configuration (`d32`, four
leaf workers) and H2 at the current optimum (`d16`, one leaf worker).
