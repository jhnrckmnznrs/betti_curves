# Changelog

All notable user-facing changes are documented here. This project follows [Semantic Versioning](https://semver.org/) and the structure of [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Maintenance

- CI freeze cleanup: remove obsolete deferred-repair wrappers, compile compact replay helpers only in tests, and name hierarchical leaf batch result types so `cargo clippy --all-targets -- -D warnings` remains a hard quality gate.
- `scripts/prepare_for_commit.sh` now mirrors the GitHub CI quality gates before the release build.

### Changed

- Hierarchical H0/H2 branch-tree history now contracts zero-persistence finalized branches online. Already-generated stale parent references are repaired at finalization time using filtration-aware strict/equality rules, so root replay retains only positive-persistence finalized history.
- Added profiling counters for input finalized events, contracted zero events, retained finalized events, and online parent-reference repairs.
### Profiling instrumentation

- Added coarse stage timings for hierarchical H0/H2 branch trees: parallel leaf batches, aggregate leaf I/O and compute, finalized-event bucketing, fan-in, root reduction, and plateau canonicalization.
- Added per-combine timings for setup, cross-interface sparsification, event sorting, pairwise union-find reduction, and parent-summary construction.
- Extended `scripts/profile_branch_tree_external.py` to aggregate the new timing counters into CSV summaries.
- No branch-tree algorithm or event-order semantics are changed by this instrumentation pass.


### Added

- Hierarchical plateau-canonical H0/H2 branch-tree modes with online binary fan-in and bounded slab-interface reconciliation state.
- Complete flat-vs-hierarchical branch-tree equivalence validator.
- Source-layout policy documenting which modules are intentionally separate and which large files should be extracted later.


### Performance

- Hierarchical leaf slabs now contract zero-persistence finalized history inline before returning their summaries. This preserves the v10 early-attach pruning while preventing diagonal leaf deaths from being materialized and rebucketed centrally. `BETTI_HIER_INLINE_LEAF_HISTORY=0` restores the v10 central-only contraction path in the same binary. Profiling reports raw/retained leaf history, leaf-local zero contractions and parent repairs, and central-history input volume.
- Hierarchical leaf summaries now finalize non-promoting boundary/internal branch deaths immediately into deferred-repair history instead of exporting them as provisional attach events. The optimization is hierarchical-only; flat/in-memory slab summaries remain unchanged as the correctness reference. Set `BETTI_HIER_EARLY_FINALIZE_LEAF_ATTACHES=0` to restore the v9 leaf behavior in the same binary. Profiling reports leaf one-boundary/internal candidates, early-finalized deaths, propagated attaches, and per-combine input attach/interface counts.
- Hierarchical H0/H2 fan-in now streams the deterministic ordered child/cross event sequence directly into the pair union-find instead of materializing a merged `Vec<Event>` at every combine. `BETTI_HIER_MATERIALIZE_FANIN=1` retains an A/B reference mode in the same binary; profiling reports `fanin_mode`, streamed-event counts, and materialized-event counts.
- Replaced hierarchical H0/H2 fan-in concatenate-plus-stable-sort with deterministic linear merging of filtration-ordered child and cross-interface streams. H0 merges five monotone streams in `(value, Attach/Interface/Cross, left-before-right)` order. H2 logically splits mixed Outside/finite attach streams and merges seven monotone streams in `(reverse value, Outside/Attach/Interface/Cross, left-before-right)` order, preserving the former stable-sort tie semantics exactly.
- Per-combine profiling now reports `merge_seconds`; `sort_seconds` is retained and is zero on the ordered fan-in path for direct comparison with previous profiles.

- Replaced the hierarchical root reducer's hash-heavy deferred-parent and same-threshold plateau state with reusable packed open-addressed integer tables. The deferred resolver now combines watched membership and parent redirects in one table, while H0/H2 plateau contraction uses a reusable generation-stamped `u64 -> u64` map with O(1) threshold reset. The flat/in-memory reducers remain unchanged as equivalence references.
- Hierarchical root replay now carries repaired parent IDs rather than full branch values through its scratch buffers; H2 Outside is represented by the existing `OUTSIDE_BRANCH_ID` sentinel. Branch birth metadata remains in compact events and exported nodes, so event ordering and tree semantics are unchanged.

- Added a hierarchical compact root-reducer fast path for prebucketed H0/H2 events. Zero-persistence compact events are replayed directly into plateau contraction and are no longer expanded into general-purpose merge objects; full merge values are materialized only for positive-persistence branches and the much smaller global interface stream.
- Replaced the hierarchical root reducer's all-local duplicate-transition HashSet with a target-driven check built only from actual global attach outcomes. This avoids inserting tens of millions of local branch transitions solely to suppress a small number of replay duplicates.
- Replaced the full `observed_merges` copy of compact local history with one repaired-parent entry per local event and reused root-reduction scratch buffers across thresholds. Deferred-parent updates retain the original threshold-delayed event order.

- Hierarchical finalized-event buckets now use compact H0/H2 records that omit the filtration value already encoded by the bucket index and flatten branch metadata into packed scalar fields. On 64-bit targets the compact records are asserted at 24 bytes; runtime profile lines report compact/full event sizes and estimated finalized-event storage.
- Hierarchical H0/H2 branch-tree reduction now stores finalized branch deaths directly in `u16` filtration buckets and passes those buckets into the root reducer, eliminating the previous second full copy of finalized local events.
- Hierarchical leaf slabs now use deterministic bounded Rayon batches. Automatic concurrency is capped at four workers and by a conservative 512 MiB concurrent-leaf budget; `BETTI_HIER_LEAF_WORKERS` and `BETTI_HIER_LEAF_BUDGET_MB` provide explicit overrides.
- Added `scripts/profile_branch_tree_external.py` for source-instrumentation-free timing, RSS, page-fault, and hierarchy-counter sweeps.

### Fixed

- Hierarchical branch-tree fan-in now separates boundary-state propagation from branch-death finalization: an older incoming H0/H2 branch can replace the elder visible on a parent boundary without killing that boundary-visible branch repeatedly at every higher fan-in level. Same-branch replay is connectivity-only, and duplicate-death diagnostics now report both competing transitions.
- GitHub-readiness gate fixes: native-F32 H2 test widening, targeted Clippy annotations for tuning-heavy kernels, and a named hierarchical H2 combine result type.
- Added `scripts/prepare_for_commit.sh` so local validation formats the tree before enforcing `cargo fmt --check`.

### Planned
- Publish thread-scaling measurements with explicit `RAYON_NUM_THREADS` capture.
- Run the 64/128-slice real-volume H2 prefix benchmark for d8 and d16.
- Publish tagged multi-platform binaries after the GitHub repository migration.

## [0.2.0] - 2026-09-05

### Added
- Disk-backed hierarchical H0 persistence with bounded pairwise interface state.
- Elder-dominated H0 attach finalization and terminal-free root reduction.
- Native 4-byte F32 key path and packed global H0 union-find state.
- Outside-aware hierarchical H2 persistence with compact births and packed state.
- Elder-dominated H2 attach pruning, direct cross-interface consumption, and outside-dominated structural pruning.
- Reproducible benchmark data, scaling plots, deterministic synthetic-data generation, and Python baseline tooling.
- GitHub Actions CI, benchmark automation, and tagged multi-platform release workflow.
- End-to-end CLI integration-test fixtures and reviewer-facing architecture documentation.
- Criterion process-level benchmark harness and Linux flamegraph helper.

### Changed
- Repository/package identity renamed from `betti_curves` to `stream_betti_curves`.
- Public presentation reframed around memory-efficient exact processing of large 3-D image volumes.
- Internal optimization notes moved out of the repository root into `docs/development/`.
- The executable remains named `betti_curves` for compatibility.

### Performance
- H0 real F32 prefix: 3792×3792×128 processed at 1.66 Mvox/s with 2.04 GiB peak RSS and 4.22 GiB peak scratch in the recorded d8 hierarchical run.
- H2 CX09T1 d16: hierarchical direct/outside-pruned path measured at 3.54 s and 20.4 MiB peak RSS versus 5.81 s and 43.2 MiB for the corresponding flat path.

## [0.1.0] - Prior public snapshot

### Added
- Initial public Rust implementation for Betti-0 and Betti-2 curves on TIFF stacks.
- Threshold-wise and event-based slabwise algorithms.

[Unreleased]: https://github.com/jhnrckmnznrs/stream_betti_curves/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jhnrckmnznrs/stream_betti_curves/releases/tag/v0.2.0
[0.1.0]: https://github.com/jhnrckmnznrs/stream_betti_curves/releases/tag/v0.1.0

## Unreleased - recursive hierarchy-history contraction (v12 experimental)

- Generalize inline finalized-history contraction from leaf slabs to every hierarchical fan-in node.
- Each internal combine now contracts zero-persistence finalized deaths and repairs retained parent references before returning its summary upward.
- Keep the 65,536-bucket centralized contractor only for the final root handoff.
- Add `BETTI_HIER_RECURSIVE_HISTORY=0` to restore the v11 central-only internal-history path for same-binary A/B validation.
- Add recursive-history profiling counters and a randomized symbolic staging validator.
