# Source layout and module-size policy

`src/` contains more modules than a small Rust CLI because the repository deliberately keeps mathematically distinct algorithms separate. The important review criterion is not the raw file count; it is whether a reviewer can locate one responsibility without traversing unrelated hot paths.

## Keep separate

The following are intentionally separate because they have different state machines or asymptotic behavior:

- threshold-wise Betti computation;
- event-based Betti computation;
- persistence H0 and H2;
- scalar/native-F32 persistence;
- disk-backed and hierarchical reconciliation;
- elder-rule branch trees;
- TIFF/scalar I/O and interface sparsification.

Merging these would create very large files with unrelated correctness invariants.

## Small support modules

Small files such as `atomic_output.rs`, `binary_io.rs`, `slab_interface.rs`, `temp_runs.rs`, and `memory_audit.rs` are kept separate because each owns a cross-cutting systems concern. They are intentionally boring and independently testable.

## Current files that deserve future extraction

The main concision issue is actually a few *large* files, not too many small files:

- `main.rs` mixes CLI parsing, preflight, and dispatch;
- `persistence_h0_scalar.rs` contains local reduction plus hierarchy;
- `persistence_h0_scalar_stream.rs` contains disk formats plus hierarchy;
- `persistence_h2_scalar.rs` and `persistence_h2_scalar_stream.rs` have the same pattern.

Those should eventually become directory modules (`cli/`, `persistence/h0/`, `persistence/h2/`) without changing public behavior. That refactor should be done separately from algorithm changes so correctness/performance diffs remain reviewable.

## Rule for new code

A new file is justified when it owns a distinct invariant or lifecycle. The hierarchical branch-tree implementation is therefore one shared module for H0 and H2 rather than four H0/H2 × memory/stream modules.
