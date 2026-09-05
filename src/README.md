# Rust source map

The crate is intentionally split by algorithmic responsibility rather than by file-size target.

- `main.rs`, `connectivity.rs`, `size_plan.rs` — CLI, mode selection, and checked planning.
- `io*.rs`, `tiff_paths.rs`, `atomic_output.rs` — TIFF input and transactional output.
- `betti*.rs`, `event_betti*.rs` — threshold-wise and event-wise Betti curves.
- `persistence_h0*`, `persistence_h2*` — persistence reducers and scalar/hierarchical variants.
- `merge_tree_*` — elder-rule branch trees; `merge_tree_hierarchical.rs` is shared by H0/H2.
- `interface_sparsify*.rs`, `local_pruning.rs`, `local_uf_state.rs`, `union_find.rs` — reusable topology kernels.
- `scalar*.rs`, `scalar_stream_tuning.rs` — exact scalar keys, ordering, and performance policies.
- `temp_runs.rs`, `binary_io.rs`, `memory_audit.rs`, `allocator_trim.rs` — systems support.

See `docs/source-layout.md` for the policy behind this split and the large files that should be extracted in a later behavior-preserving refactor.
