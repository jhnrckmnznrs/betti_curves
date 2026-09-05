# v1.26.1 real-F32 H2 prefix profiling

This harness measures the optimized hierarchical H2 path on real F32 TIFF
prefixes without copying source slices. It is intended to choose between d8 and
d16 before a full-volume H2 run.

The default candidate stack is:

- native32 F32 scalar keys;
- compact local and global H2 birth state;
- packed global/pairwise UF;
- direct local event storage;
- elder-dominated attach pruning;
- direct cross-interface consumption;
- outside-dominated structural pruning;
- terminal-free final reduction.

Peak temporary storage is sampled from a private `BETTI_TEMP_DIR` subtree. The
profiler writes its CSV after every successful run so a later failure does not
lose earlier measurements.

The order-independent sum/xor SHA-256 fingerprints are diagnostics for large
prefix runs, not a formal substitute for exact Counter-based equivalence. The
CX09T1 validation gates remain the correctness oracle.
