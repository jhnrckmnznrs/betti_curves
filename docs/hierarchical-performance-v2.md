# Hierarchical branch-tree performance v2

This pass makes two performance changes while preserving the existing hierarchical H0/H2 CLI modes and plateau-canonical output contract.

## A. Single-copy finalized-event storage

Finalized H0/H2 branch deaths are inserted directly into 65,536 `u16` filtration buckets during leaf ingestion and hierarchical fan-in. The root reducer consumes these buckets by ownership. It no longer copies a flat vector of all finalized deaths into a second threshold-bucketed representation.

The event order inside each threshold bucket is deterministic: leaves are committed in slab-ID order, fan-in is serial and deterministic, and final events are appended at the same logical point at which the former flat `finalized` vector was appended.

## B. Bounded deterministic leaf parallelism

Leaf slabs are processed in Rayon batches. Completed leaves are sorted by slab ID before they enter the existing serial binary-counter fan-in.

Default worker selection is bounded by:

- the current Rayon thread pool;
- a hard default maximum of 4 leaf workers;
- a conservative 512 MiB concurrent-leaf memory budget using a 128-byte/voxel working-set estimate.

Override controls:

```bash
BETTI_HIER_LEAF_WORKERS=1
BETTI_HIER_LEAF_BUDGET_MB=256
```

An explicit worker count is still capped by the Rayon pool size. `RAYON_NUM_THREADS` can therefore provide a second hard ceiling.

## Required validation

Build first:

```bash
cargo fmt
cargo test --release
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
```

Then run exact flat-vs-hierarchical equivalence:

```bash
python3 validation/check_branch_tree_hierarchical_equivalence.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 8 \
  --foreground-connectivity 26 \
  --dimensions h0 h2
```

Also check the serial scheduling baseline:

```bash
BETTI_HIER_LEAF_WORKERS=1 \
python3 validation/check_branch_tree_hierarchical_equivalence.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 8 \
  --foreground-connectivity 26 \
  --dimensions h0 h2
```

## Profiling matrix

The external profiler requires no source instrumentation:

```bash
python3 scripts/profile_branch_tree_external.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depths 8 16 32 \
  --foreground-connectivity 26 \
  --repeats 3 \
  --output profiles/cx09t1_hierarchy_v2.csv
```

For worker scaling, repeat the same command with:

```bash
BETTI_HIER_LEAF_WORKERS=1 ...
BETTI_HIER_LEAF_WORKERS=2 ...
BETTI_HIER_LEAF_WORKERS=4 ...
```

Compare median compute time and peak RSS. Do not raise the default worker cap until the large-volume memory tests establish that the extra concurrency is safe.
## C. Compact finalized-event records (v2.1)

The hierarchical-only finalized-event buckets now store compact records rather than the general merge-event structs. The bucket index already identifies the `u16` threshold, so storing that value again in every event was redundant. H0 stores child/parent IDs and births directly; H2 uses the existing `OUTSIDE_BRANCH_ID` sentinel for the outside parent and otherwise stores the finite parent ID/birth directly.

The general flat and slab-local merge-event types are unchanged. Compact records are expanded back to the exact original event representation only when the root reducer visits their threshold bucket. Deferred-parent repair therefore receives the same branch IDs and birth values as before.

On 64-bit targets the compact H0 and H2 records are tested to occupy 24 bytes and to be smaller than their corresponding general event types. The hierarchical profile line now includes:

```text
compact_event_bytes=...
full_event_bytes=...
finalized_storage_bytes=...
```

The external profiling summary also reports the median finalized-event count and compact storage estimate. Exact flat-vs-hierarchical equivalence remains the required correctness gate after this storage-only optimization.


## D. Packed root-state replay (v2.3)

After compact finalized-event storage, profiling on CX09T1 showed the root
reducer remained the dominant H0/H2 stage. The hierarchical prebucketed path
therefore uses packed integer state without changing event order:

- deferred-parent repair combines watched membership and redirect state in one
  deterministic open-addressed table keyed by finite branch ID;
- H0/H2 same-threshold diagonal contraction uses a reusable open-addressed
  `u64 -> u64` table with generation stamps, so clearing between thresholds is
  O(1);
- repaired-parent scratch buffers store only parent IDs rather than full branch
  values; H2 Outside uses `OUTSIDE_BRANCH_ID`;
- the flat/in-memory reducers retain their HashMap/HashSet implementation and
  continue to serve as an independent exact-equivalence reference.

Birth metadata is not removed from compact events or exported nodes. It is
omitted only from packed redirect state because root replay uses redirects for
identity resolution, duplicate suppression, and parent contraction; none of
those operations compare parent birth values.

The required correctness gate remains:

```bash
BETTI_HIER_LEAF_WORKERS=1 \
python3 validation/check_branch_tree_hierarchical_equivalence.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 8 \
  --foreground-connectivity 26 \
  --dimensions h0 h2
```

Profile the root-state change with one worker first so leaf scheduling does not
obscure `root_reduce_seconds`, then repeat H0 with four leaf workers once exact
equivalence is established.


## E. Sort-free ordered hierarchical fan-in (v2.4)

Hierarchical H0/H2 child summaries already emit attach and interface events in
filtration-value order, and `sparsify_cross_interface` guarantees that retained
cross-interface edges are emitted in filtration order. The previous fan-in
nevertheless concatenated those streams and ran a stable comparison sort at
every combine.

The ordered fan-in path now performs a linear merge while preserving the old
stable-sort key exactly:

- H0: ascending value, then Attach, Interface, Cross; left child precedes right
  child when value and event kind tie.
- H2: descending value, then Outside, finite Attach, Interface, Cross; left
  child precedes right child when value and event kind tie. Because H2 attach
  vectors may mix Outside and finite attaches at one threshold, the merge
  logically splits those two priority classes before merging.
- Cross-interface events retain their original emission order for equal values.

Production fan-in no longer calls `sort_by`. Unit tests reconstruct the former
concatenate-plus-stable-sort sequence on tied synthetic events and require the
linear merge output to be exactly equal.

Per-combine profiling keeps `sort_seconds=0` for backward compatibility and
adds `merge_seconds`. The external profiling script reports
`median_combine_merge_seconds` so v7 can be compared directly with v6's
sorting cost.
