# U16 H0 persistence: bounded parallel leaf preparation

The native-U16 hierarchical H0 persistence pipeline previously prepared slabs serially even though each leaf sweep is independent until hierarchical handoff. On CX09T1 at slab depth 16, the native32 v20 profile attributed about 5.15 seconds to summed local leaf sweeps, making leaf preparation the dominant H0 persistence cost.

v21 prepares bounded batches of independent leaves with Rayon. Each worker owns its slab state, attach/interface run writers, and a small in-memory vector of positive finalized pairs. After the batch finishes, the coordinator sorts leaves by slab ID, writes those positive pairs to the global pair stream in deterministic order, and performs the existing binary-counter hierarchical fan-in serially.

This preserves the filtration result because leaf slabs have no cross-slab edges until fan-in. Parallel scheduling changes neither each leaf's deterministic event order nor the slab-order handoff into the hierarchy.

Configuration:

- `BETTI_PERSIST_H0_LEAF_WORKERS=<N>` requests a positive worker count, capped by the Rayon pool.
- `BETTI_PERSIST_H0_LEAF_BUDGET_MB=<MiB>` bounds the default automatic worker choice; default 512 MiB.
- Without an explicit worker count, at most four workers are used.

The H2 persistence path is intentionally unchanged in v21 because previous measurements showed much weaker scaling from leaf parallelism there.
