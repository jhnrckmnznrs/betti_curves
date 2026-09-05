# Hierarchical finalized-history contraction

This note states the invariant used by the branch-tree hierarchical history
contraction introduced after the ordered-fan-in implementation.

## Setting

A hierarchical summary exports only state that is still visible on its parent
boundary: boundary branch seeds, attach transitions, and interface merges. A
`Final` branch transition is emitted only when the dying branch no longer has
a parent-boundary representative, or when an incoming internal branch dies
into the boundary-visible elder. Consequently the dying branch of a `Final`
transition is absent from the parent summary's live branch state.

Call a finalized transition

\[
    c \xrightarrow[d_c]{} p
\]

where `c` is the dying branch, `p` is the surviving parent branch, and `d_c`
is the filtration value at which the transition becomes final.

## No-future-reference invariant

**Invariant.** Once `c -> p` is emitted as `Final` by a hierarchical node, no
transition generated strictly above that node can newly name `c` as its live
parent branch.

The reason is structural: a branch can reach a higher hierarchical level only
through parent-boundary state. `Final` is precisely the case in which the
dying child is not propagated as that state. Stale references to `c` may
already exist in finalized history generated below the node, which is why
parent repair is still necessary, but those references are all already known
when `c` becomes final.

## Filtration-time repair rule

For H0 the filtration is sublevel and root replay is increasing in value. If
a retained finalized event `e` has provisional parent `c`, then after the true
death `d_c` of `c` becomes known:

* if `c` has positive persistence, redirect `e.parent` to `p` exactly when
  `death(e) > d_c`;
* if `c` has zero persistence (`birth(c) = d_c`), redirect also at equality,
  so every reference with `death(e) >= d_c` contracts through `c`.

The strict inequality in the positive case matches deferred root replay: the
redirect caused by a death at threshold `d_c` becomes visible only after that
threshold has finished. Equality for a zero-persistence branch is instead the
plateau-contraction rule used by the tree recorder.

For H2 the filtration order is reversed. Therefore:

* a positive branch redirects references with `death(e) < d_c`;
* a zero-persistence branch also redirects equality, i.e. `death(e) <= d_c`.

## Consequence

A zero-persistence finalized branch can be contracted at the hierarchical node
where it becomes final. Its own death record never needs to reach root replay.
All already-generated retained references to it are redirected immediately,
and by the no-future-reference invariant no new higher-level reference can be
created afterward.

A positive-persistence finalized branch must remain as an exported branch-tree
node. Its death record is retained, but stale parent references that occur
after its true death in filtration order are repaired immediately.

Thus root replay needs only positive-persistence finalized history plus the
still-live root/interface events. The former implementation retained every
voxel-level finalized transition, most of which were diagonal plateau events.

## Correctness boundary

This contraction does **not** alter:

* leaf or pair union-find event order;
* the boundary-visible attach rule;
* interface or cross-interface summaries;
* root global union-find mutation order;
* positive-persistence branch deaths;
* essential/outside roots.

The optimization changes only how already-final finalized history is stored
and how stale parent IDs in that history are repaired before root replay.
Exact flat-vs-hierarchical plateau-canonical equivalence remains the required
end-to-end oracle.
