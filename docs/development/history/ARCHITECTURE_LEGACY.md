# Architecture and invariants

This document is the map for reviewers and future contributors. The central
rule is simple: an optimization is acceptable only when it preserves the
connected-component partition at every filtration value, not only the final
component count.

## 1. Data flow

For one TIFF stack, a typical event or persistence path is:

1. `main.rs` parses the command, selects dual connectivity, opens the stack,
   and runs the checked `SizePlan`.
2. `io.rs` or `io_scalar.rs` validates the TIFF stack and reads z-blocks.
3. A mode module processes each block in filtration order and records local
   births, successful unions, interface representatives, and outside events.
4. `interface_sparsify.rs` replaces a full cross-face graph by a forest that
   preserves every prefix partition.
5. A global reducer replays local and cross-slab events with the elder rule.
6. An atomic writer commits the final CSV or JSON file.

The threshold-wise `betti0.rs` and `betti2.rs` paths deliberately use a simpler
calculation. They are slower but provide a useful internal comparison.

## 2. Non-negotiable topology rules

All modes must use exactly the same conventions:

- foreground: `I <= t`;
- strict background: `I > t`;
- dual connectivity: foreground/background is 6/26 or 26/6;
- ordinary non-periodic box;
- every boundary-touching background component joined to one virtual outside;
- half-open intervals `[birth, death)`;
- no positive-length interval may be lost because two events have the same
  value;
- `-0.0` and `+0.0` are the same scalar value;
- all other finite scalar values keep their exact numerical order.

When adding a mode, first state which existing mode it must agree with. Compare
intervals as multisets because a streaming writer need not globally sort them.

## 3. Module map

| Module | Responsibility |
|---|---|
| `connectivity.rs` | Connectivity and CLI mode parsing |
| `tiff_paths.rs` | Natural slice order and recursive stack discovery |
| `io.rs` | U8/U16 stack validation and block reads |
| `io_scalar.rs` | U8/U16/F32/F64 stack validation and block reads |
| `scalar.rs` | Exact monotone scalar keys and signed-zero rule |
| `scalar_order.rs` | Counting and radix ordering |
| `size_plan.rs` | Checked products and identifier limits before computation |
| `slab_interface.rs` | Deterministic boundary-node numbering and sentinels |
| `union_find.rs` | Local and dynamic disjoint-set structures |
| `local_pruning.rs` | On-demand active-neighborhood component pruning and bounded cache |
| `interface_sparsify.rs` | Prefix-preserving cross-face forest |
| `interface_sparsify_scalar.rs` | Scalar wrapper around the same forest rule |
| `event_betti*.rs` | Direct Betti-curve event summaries and reducers |
| `persistence_*.rs` | Elder-rule interval summaries and reducers |
| `merge_tree_*.rs` | Elder-rule branch recording and reducers |
| `event_scalar_common.rs` | Shared scalar event records and run readers |
| `temp_runs.rs`, `binary_io.rs` | Temporary run lifecycle and binary records |
| `atomic_output.rs`, `csv.rs` | Transactional final outputs |

The H0 and H2 mode modules contain similar-looking hot loops. This duplication
is intentional where the mathematical direction, outside state, or concrete
record layout differs. Share a helper only when its full contract is identical
for every caller.

## 4. Filtration direction

H0 activates foreground voxels in increasing value order. An adjacency edge
has value `max(I(u), I(v))`. At a successful union, the component with the
smaller `(birth value, birth identifier)` pair survives.

H2 is computed through decreasing strict-background components. A background
edge has value `min(I(u), I(v))`. The explicit outside root is older than every
finite background component. A background branch born at foreground death
`d` and joined at foreground birth `b` produces `[b, d)`.

Equal-valued events may change the recorded elder-rule parent chain, but they
must not create or remove positive-length intervals. Any refactor of event
ordering needs the same-level multi-component regression test.

## 5. Neighborhood pruning and its claim boundary

`NeighborhoodComponentPruner` records the already-active neighbors of a newly
activated voxel as a bit mask. It keeps one neighbor from each connected
component of the induced active-neighborhood graph. The omitted edge is safe
because its endpoint already has an earlier active path to the retained
representative. This depends on processing one voxel at a time; pre-activating
an equal-value group breaks the invariant.

For 26-connectivity, a 65,536-slot direct-mapped cache stores repeated answers.
A miss is computed at run time, and a collision only causes recomputation. For
6-connectivity, the six face neighbors are pairwise disconnected within the
six-neighbor rule, so the input mask is already minimal and no cache is
allocated.

This helper is not the precomputed cubical lower-star table described by
[Le Breton, Szustakowski, and Piraud](https://arxiv.org/abs/2606.04801). Their
table contains cubical cell classifications, zero-persistence pair data, and
local tie order. This helper contains only graph component representatives.
See [`RELATED_WORK.md`](RELATED_WORK.md) before changing this distinction.

## 6. Slab summaries

A slab summary contains enough information to replay the component evolution
that can affect another slab:

- successful internal birth and merge events;
- a representative for each component meeting a lower or upper face;
- attachments from interior branches to a visible representative;
- outside flags or outside events for H2;
- a prefix-preserving forest among visible representatives.

An interior component that never reaches a face can be finalized locally. A
component that reaches a face stays live until global reconciliation decides
its fate.

Boundary IDs are computed, not stored per voxel. For a slab with depth greater
than one, the lower face owns IDs `0..A` and the upper face owns `A..2A`.
One-slice slabs use one face because the two faces are identical. The sentinel
`u32::MAX` means “no interface representative” and must never be indexed or
emitted as a real outside node.

## 7. Exact cross-face reduction

For 6-connectivity, only matching `(x, y)` sites across the two faces are
adjacent.

For 26-connectivity, a face site can see the 3-by-3 neighborhood on the other
face. Materializing every candidate requires

```text
C26 = (3 * width - 2) * (3 * height - 2)
```

edge records. The revised algorithm instead orders the `2A` vertices by their
activation value. When a vertex activates, it visits cross-neighbors that are
already active. A union-find keeps only successful joins. This is Kruskal's
algorithm expressed as vertex activation, because a cross-edge becomes
available exactly when its later endpoint activates.

The retained forest must satisfy this test for every threshold prefix:

```text
components(full cross graph at t) == components(retained forest at t).
```

This equality is stronger than preserving the final connected components and
is what makes the reduction safe for Betti curves and persistence.

## 8. Numeric ordering

`ScalarKey` maps finite F32 and F64 values to unsigned keys whose integer order
is the numerical order. Both zero bit patterns map to one key. The source pixel
type remains attached to the stack so ordering can use:

- 256-bucket counting order for U8;
- 65,536-bucket counting order for U16;
- stable eight-pass least-significant-byte radix order for F32/F64 keys.

Do not replace exact keys with lossy conversion to `f64`, string ordering, or
an approximate comparator.

## 9. Resource model

The size plan checks local `u32`, global `u32`, and compact interface `i32`
limits. It is a representation check, not a memory promise.

Disk-backed modes bound selected event or interval buffers by writing sorted
runs. They still retain global interface union-find state proportional to `G`,
and some current merges keep one reader per run. New “streaming” claims must
name separately:

- peak memory;
- temporary disk bytes;
- maximum open files;
- whether restart after interruption is supported.

## 10. Output lifecycle

Final structured outputs use `AtomicOutput` or the matching Python context
manager. A writer creates a unique same-directory partial file, writes and
flushes it, synchronizes it, renames it over the destination, and synchronizes
the parent directory on Unix. If writing fails, the old destination remains
and the partial file is removed.

Temporary binary runs are intentionally different: they are internal scratch
state owned by `TempRunDirectory` and are removed with that directory.

## 11. Safe extension points

- Add an input format behind a future `VolumeSource` interface; preserve
  checked block and halo reads.
- Add a reducer by consuming existing typed summary records; do not reinterpret
  their filtration direction.
- Add a numeric storage type by defining its exact order and zero/nonfinite
  policy first.
- Add a comparison metric in the Python utility only after defining its domain,
  units, normalization, and stability claim.
- Add canonical plateau nodes as a separate tree representation; do not relabel
  the current elder-rule branch file as canonical.

The main research-level extension is a composable two-face summary with an
associativity proof. Until that proof and exhaustive small-volume tests exist,
do not claim bounded global memory by discarding older interface state.
