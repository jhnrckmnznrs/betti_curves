# Relation to Flash Cubical

This note records the boundary between this repository and the June 2026
preprint by Titouan Le Breton, Karol Szustakowski, and Marie Piraud,
[*Fast Cubical Persistent Homology on 2D and 3D Images via Union-Find,
Pruning, and Lookup Tables*](https://arxiv.org/abs/2606.04801), together with
its [Flash Cubical repository](https://github.com/T-prog123/FlashCubical).
The comparison is against preprint version 1, submitted on June 3, 2026.

The present project began before its authors learned of that preprint. The
current documentation nevertheless treats the preprint as prior public work.
This development history is not used to claim an earlier public result.

## Shared ideas that are not claimed as new

Both projects use established component-persistence ideas:

- union-find for degree-zero persistence;
- duality and a virtual outside vertex for top-dimensional persistence;
- local bit masks on regular image neighborhoods; and
- removal of work that cannot change the requested persistence output.

These shared ideas require citation and a clear comparison. They do not by
themselves show that the two implementations use the same algorithm.

## Different mathematical and software roles

| Question | This Rust repository | Flash Cubical v1 |
|---|---|---|
| Main output | 3D digital H0 and dual H2 curves, intervals, and elder-rule branch data | Full 2D/3D V-filtration persistence, including 3D H1 |
| Large-data plan | Slabs, interface summaries, prefix-preserving cross-face forests, and optional disk runs | No slabwise or out-of-core reconciliation theorem is described in v1 |
| Meaning of a local mask | Active graph neighbors of one voxel | Cells in a cubical lower star, including local pairing and tie-order data |
| Stored acceleration data | A bounded direct-mapped cache; misses are computed at run time | Lookup artifacts computed in advance and loaded by the package |
| Cross-face reduction | Runtime Kruskal forest between two slab faces | Not the local lower-star lookup operation described in v1 |
| Middle homology | Not computed | A pruned matrix reduction computes 3D H1 |

The Rust type `NeighborhoodComponentPruner` returns one already-active
neighbor from each connected component of the active induced neighborhood.
Its correctness condition is that all edges among earlier vertices have
already been processed. It does not classify cubical cells, determine a
second cell order, identify H1 pairs, or load a generated table artifact.

The interface sparsifier has a different job. It sees only two neighboring
slab faces and keeps a Kruskal forest that preserves the component partition
at every filtration value. It reduces temporary cross-face storage; it is not
a lower-star cell-pair classifier.

## Source and output checks made during this revision

- No Flash Cubical source file or binary lookup artifact is included here.
- A normalized lexical scan found no shared sequence of eight source tokens
  between this Rust `src/` tree and the Flash Cubical C++ tree. This is a
  screening check, not a proof of authorship or priority.
- On a fixed random `4 x 4 x 4` array with distinct U16 values, the two
  programs produced the same positive-length V-filtration H0 intervals; both
  produced no H2 interval.
- On a `3 x 3 x 3` array with a low shell and a high center, both produced the
  H2 interval `[0,1)`.

The output checks confirm the intended common mathematical target in these
small cases. They are not a performance benchmark and do not test H1. A fair
timing or memory comparison must match the cubical construction, requested
homology dimensions, input type, tie convention, hardware, compiler settings,
and whether table-building time is included.

