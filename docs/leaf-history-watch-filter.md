# Leaf-history watch filter

## Motivation

The hierarchical branch-tree leaf reducer finalizes many branches that never
survive beyond the slab. After the compact/deeper-leaf changes, a representative
CX09T1 d16 run observes roughly 29 million leaf-local finalized events per
homology dimension, while fewer than 0.2 million positive-persistence events
survive. The generic local history contractor previously queried a `HashMap`
for every finalized child to determine whether a retained positive event used
that child as a provisional parent.

That query is unnecessary when no retained event can reference the child.
Because finite leaf branch identities are slab-local voxel IDs, the set of
currently referenced provisional parents can be represented exactly by one bit
per local voxel.

## Invariant

Let `W` be the set of finite branch IDs that occur as the provisional parent of
at least one retained positive-persistence leaf event. The implementation keeps

```text
watch_bit[c] = 1  iff  c is a key in by_parent.
```

Before repairing the death of branch `c`, the fast path checks `watch_bit[c]`.
If the bit is zero, `by_parent[c]` is provably absent and the hash lookup can be
skipped. If the bit is one, the existing filtration-aware repair logic is run
unchanged.

For H0, a retained event pointing to `c` is redirected when the child's death
is zero-persistence or when the retained event dies strictly later. For H2 the
strict inequality is reversed. The distinguished H2 Outside branch is never a
finite watched key.

The filter is therefore an exact negative-membership cache, not a probabilistic
Bloom filter: it cannot produce false negatives.

## Compact local events

The hot leaf union path also keeps finalized events as slab-local records:

```text
(value: u16, child_local: u32, child_birth: u16, parent_local: u32)
```

Full global branch IDs and parent birth values are reconstructed only for the
small retained history returned by the leaf. This avoids constructing full
H0/H2 branch objects for diagonal events that are immediately discarded.

## A/B control

The watch filter is enabled by default. Set

```bash
BETTI_HIER_LEAF_HISTORY_WATCH_FILTER=0
```

to force the compact contractor to perform the reference hash lookup for every
finalized event. This disables only the negative-membership shortcut; the
compact local-event representation remains enabled.

The profiling output reports watch checks, skipped lookups, actual hash
lookups, and zero-persistence fast drops.
