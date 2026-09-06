## v23.6 frozen checkpoint — U16 H2 production profile

- Froze the confirmed native-U16 H2 production profile described in `V23_6_FREEZE.md`.
- Corrected `V23_6_RUN_COMMANDS.txt` to include the mandatory compact local/global H2 birth-state and packed global-UF flags plus the benchmarked direct-cross/outside-dominated hierarchy settings.
- Added the CX09T1 frozen output under `benchmarks/data/v23_6_freeze/` with SHA-256 `6f08ac4bf5b2d155724d28b1d23dd6c4697ff53f707e5fd4786795f63a3744e6`.
- No Rust algorithm change relative to the v23.6 production-consolidation code.

## v23.6 production consolidation — promoted U16 H2 fast interface query

- Promoted the v23.5 fast root-invariant interface query to the native-U16 H2 production default after the pre-registered v23.5.2 blocked confirmation passed all timing gates.
- The 5×7 d16 confirmation produced a median wall-time delta of **-1.52%**, a median local-sweep delta of **-1.75%**, **29/35** candidate wall-time wins, exact two-sided sign-test **p=0.000116842**, and negative wall medians in **4/5** blocks and both run-order strata.
- `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY` is now default-on. Set it to `0`, `false`, `off`, or `no` to restore the exact division/remainder reference path for regression/ablation work.
- Kept native-U16 H2 root dedup opt-in-only, `parent-two-hop` experimental/rejected for production, and `parent-cached-find` experimental/neutral; the production neighbor-root strategy remains `parent-shortcut`.
- Preserved the complete v23.5.2 blocked confirmation under `benchmarks/data/v23_5_2_confirmation/`, including the decision file and raw paired/block summaries. Hardware `perf` counters were unavailable on that host and were not a promotion requirement.
- Added a production-default regression gate that verifies: unset fast-interface environment => ON; explicit `=1` => ON; explicit `=0` => reference OFF; and exact ordered persistence equivalence between all three states on the representative U16 stack.
- Moved historical v23.2-v23.5.1 root-level run-command sheets into `docs/development/run_commands/`; only the current `V23_6_RUN_COMMANDS.txt` remains at repository root.
- Kept the Cargo package version at `0.2.0` for this internal freeze checkpoint. The tree also contains other unreleased feature-level work, so the next public semantic-version tag should be chosen only when that broader release scope is reviewed.

## v23.5.2 experimental — blocked d16 confirmation with optional perf counters

- Kept the v23.5 fast-interface-query Rust implementation unchanged and preserved `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=0` as the source default.
- Recorded the v23.5.1 31-pair d16 confirmation as inconclusive: its paired median still favored the candidate, but pair-to-pair wall/sweep variability was much larger than the expected ~1–2% effect and the pre-registered win/sign-test gate did not pass.
- Replaced one long sequence with a pre-registered blocked design: five blocks, seven measured pairs per block, one warmup pair per block, alternating A/B order, and a 10 s cooldown between blocks.
- Added optional `perf stat` collection (`cycles`, `instructions`, `branches`, `branch-misses`) plus `/proc/loadavg` and available CPU-frequency snapshots. `perf` is auto-probed; lack of permission does not invalidate the timing experiment.
- The timing promotion gate requires: 35 pairs in a 5x7 design; negative median wall and local-sweep deltas; at least 24/35 wall wins; exact two-sided wall sign-test p<0.05; at least four of five block wall medians negative; and negative wall medians in both order strata. Perf counters are corroborating evidence, not a hard requirement.
- Added `scripts/profile_u16_h2_fast_interface_query_blocked.py`, `scripts/evaluate_u16_h2_fast_interface_query_blocked.py`, and `scripts/confirm_u16_h2_fast_interface_query_v23_5_2.sh`.
- Preserved the complete v23.5.1 confirmation outputs under `benchmarks/data/v23_5_1_confirmation/`.

## v23.5.1 experimental — focused d16 confirmation gate

- Kept the v23.5 fast-interface-query implementation unchanged and preserved `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=0` as the source default pending confirmation.
- Recorded the first 11-pair/depth A/B result as promising: d16 median wall/local-sweep deltas were about -1.43%/-2.09%, with 9/11 wall wins; d32 and d64 also pointed in the same direction.
- Added a pre-registered d16-only confirmation protocol with 31 alternating reference/candidate pairs and no sweep diagnostics.
- The promotion gate requires: exact d16/d32/d64 persistence equivalence; negative d16 median wall and local-sweep deltas; at least 23/31 non-tied wall wins; exact two-sided wall sign-test p<0.05; and negative median wall deltas in both run-order strata.
- Added `scripts/confirm_u16_h2_fast_interface_query_v23_5_1.sh` and `scripts/evaluate_u16_h2_fast_interface_query_confirmation.py`. The evaluator records a recommendation only; it does not flip production defaults.
- Preserved the initial v23.5 paired CSVs under `benchmarks/data/` as pre-confirmation evidence.

## v23.5 experimental — U16 H2 fast root-invariant interface query

- Interpreted the v23.4 sampled three-region timing split as timer-overhead dominated and retained only its exact operation counters as optimization evidence.
- The d16/CX09T1 exact audit recorded 71,472,296 interface-representative queries, making root-invariant interface classification a substantially larger target than the previous sub-million zero-hop-find candidate.
- Added `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=1`, which replaces per-query division/remainder with a precomputed upper-face offset plus lower/upper range comparisons and subtraction while preserving the root-invariant interface semantics. The reference/default remains off until A/B validation passes.
- Added a Rust equivalence unit test, an independent randomized arithmetic model, exact d16/d32/d64 persistence equivalence, and an interleaved paired benchmark.
- Preserved the v23.4 structural CSVs under `benchmarks/data/`.

## v23.4 experimental — U16 H2 structural sweep profiling

- Kept the production U16 H2 leaf algorithm unchanged: slab depth 16, `parent-shortcut`, and native-U16 root deduplication off.
- Recorded the v23.3 cached-parent result as a neutral/depth-specific micro-optimization rather than promoting it: d16 remained slightly slower, d32 showed a small suggestive gain, and d64 was neutral.
- Added deterministic hashed ~1/1024 voxel sampling under `--sweep-diagnostics` to time three contiguous local-sweep regions without calling `Instant::now()` for every voxel: activation/boundary/interface bookkeeping; neighborhood construction/pruning; and union/persistence/event emission.
- Added exact U16 hierarchical sweep aggregates for active-state checks/hits, representatives, pruning-cache behavior, component-mask computations, interface-state operations, and final/attach/outside/interface event counts.
- Added `scripts/profile_u16_h2_sweep_structure.py` and `scripts/check_u16_h2_sweep_structure.sh` for a one-run d16/CX09T1 structural diagnosis.
- Preserved the v23.3 paired and diagnostic CSVs under `benchmarks/data/` as optimization provenance.

## v23.3 experimental — U16 H2 cached-parent fallback find

- Kept `parent-shortcut` as the production/default H2 root-carrying path; v23.2 diagnostics showed that unconditional extra-hop checks are not profitable.
- Added `--neighbor-root-check parent-cached-find`, which reuses the first parent word already loaded by the direct-parent probe when a fallback find is necessary.
- Added `LocalUnionFindState::find_from_known_parent`, preserving the same path-halving updates and traversed-edge count as ordinary `find` while avoiding a duplicate initial parent/root load.
- Root neighbors discovered by the parent probe bypass the zero-hop `find` entirely.
- Added randomized exact path-halving-state validation, d16/d32/d64 persistence equivalence, and an interleaved paired profiler with a separate diagnostics pass.
- Root dedup remains opt-in-only and is forced off in this experiment.

## v23.2.1 experimental — two-hop profiler diagnostics fix

- Corrected the v23.2 profiling harness: paired timing remains intentionally uninstrumented, while path-depth/two-hop counters are now collected in a separate `--sweep-diagnostics` pass so diagnostics cannot perturb the timing comparison.
- Added `scripts/diagnose_u16_h2_two_hop_parent_shortcut.py` for a cheap diagnostic-only rerun after an existing timing experiment.
- The original v23.2 timing CSVs remain valid for performance comparison; only their zero-valued diagnostic counters were non-informative.

## v23.2 experimental — U16 H2 two-hop parent shortcut

- Restored native-U16 H2 root deduplication to opt-in-only after the paired v23.1 CX09T1 benchmark showed reproducible wall/local-sweep regressions despite eliminating about 66% of nominal union attempts.
- Added `--neighbor-root-check parent-two-hop`: after the existing current-root/direct-parent checks miss, inspect one grandparent link before falling back to `find(neighbor)`. The production default remains `parent-shortcut`.
- Added exact two-hop support for both parent-rank and packed local union-find layouts without path compression in the shortcut itself.
- Added neighbor-find path-depth diagnostics and same-root fallback depth buckets, plus two-hop check/hit counters in the native-U16 H2 leaf profile.
- Added d16/d32/d64 exact persistence equivalence validation and an interleaved 11-pair/depth reference-vs-candidate profiler with robust paired statistics.
- Preserved the v23.1 paired root-dedup CSVs under `benchmarks/data/` as negative optimization provenance.

## v23 experimental — U16 H2 exact persistence root deduplication

- Native-U16 hierarchical H2 persistence now resolves representative-neighbor UF roots before center-edge insertion and retains only the first representative of each distinct pre-existing root.
- Duplicate-root edges are exact same-component no-ops; first-occurrence order is preserved, so persistence action ordering and Outside/interface semantics are unchanged.
- Added unconditional hot-path union counters for this experiment, fixing the v22 profile's zero-valued union diagnostics when expensive sweep diagnostics are disabled.
- `BETTI_PERSIST_H2_ROOT_DEDUP=0` restores the v22 no-dedup path in the same binary.
- Added a randomized symbolic model, exact interval-multiset validator, dedicated d16/d32 A/B profiler, and `docs/u16-h2-root-dedup-persistence.md`.
- Reworked the v23 performance gate into an interleaved paired d16/d32/d64 design (11 pairs/depth by default) with median/MAD/IQR paired deltas, wall-time CV, and per-pair invariant checks; added a multi-depth exact equivalence/slab-invariance gate and `scripts/check_u16_h2_root_dedup_candidate.sh`.
- The grouped OFF-then-ON d16/d32 profile is retained only as exploratory evidence and is not sufficient for a production freeze.

## v22 experimental — U16 H2 plateau-zero persistence elision

- Native-U16 hierarchical H2 persistence now discards finite `[t,t)` pairs at the union decision before constructing `FinitePair` or `LocalPersistenceAction` values.
- The union-find state transition, positive intervals, attach/interface events, and Outside semantics are unchanged.
- `BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION=0` restores the v21 materialize-and-filter path for exact same-binary A/B validation.
- Added a randomized symbolic model, exact interval-multiset validator, dedicated d16/d32 profiler, and `docs/u16-h2-plateau-zero-persistence.md`.

## v20.1 experimental

- Remove the unused `z0` field from `U16ScalarBlock` so the native-U16 persistence path remains clean under `cargo clippy -- -D warnings`.


## Unreleased — U16 hierarchical persistence candidate

### Experimental v20: native U16 persistence keys

- Added four-byte exact `U16Key` storage for hierarchical U16 H0/H2 persistence, with `65536` reserved as the compact H2 Outside marker.
- U16 hierarchical persistence now uses native32 keys by default; `BETTI_PERSIST_U16_NATIVE_KEYS=0` restores the v19 wide64 path for same-binary A/B validation.
- Added exact native32-vs-wide64 interval-multiset validation and a dedicated benchmark script.
- Added leaf counters for zero-persistence finite pairs that are filtered before output; v20 measures this workload but does not alter plateau semantics.

- Extend `h0-scalar-hierarchical-stream` and `h2-scalar-hierarchical-stream` to non-F32 scalar stacks, including native U16 TIFF input, while preserving the existing F32/native32 path.
- Reuse decoded scalar slab storage directly in the hierarchical leaf reducer and emit finalized pairs / attach / interface / outside events through monotone direct sinks rather than materializing per-leaf event vectors.
- Replace the legacy U16 flat cumulative-interface reconciliation with bounded pairwise hierarchical fan-in when the hierarchical modes are selected.
- Keep the existing `h0-scalar-stream` / `h2-scalar-stream` implementations as independent exact references for equivalence testing.
- This candidate intentionally keeps the canonical 64-bit `ScalarKey` for U16; native-width integer keys are a follow-up optimization after the hierarchy transfer is measured.
# Changelog

## v18 freeze checkpoint — H2 exact UF-root deduplication

- Promoted exact fixed-array H2 UF-root deduplication to the production default.
- Changed the lazy 3x3x3 shell-component pruner to opt-in only (`BETTI_HIER_H2_SHELL_PRUNE=1`) after the controlled d16 A/B showed substantial slowdown from extra shell-state checks.
- Preserved `BETTI_HIER_H2_ROOT_DEDUP=0` as the exact same-binary reference path.
- Recorded the freeze benchmark and decision in `docs/h2-root-dedup-freeze.md`.
- No H0 algorithm changes.

## v18 experimental — H2 shell-component pruning and exact root deduplication

- Added an H2-only lazy 3x3x3 shell pruner for six-connected face neighbors. It retains one face edge per already-active shell component and never scans shell voxels unless multiple face candidates are active.
- Added exact fixed-array UF-root deduplication before H2 union/action processing. For six-connectivity at most six pre-existing roots are collected (the fixed buffer also supports the 26-connected mode); duplicate roots are skipped before the union kernel.
- Added independent A/B switches: `BETTI_HIER_H2_SHELL_PRUNE=0` and `BETTI_HIER_H2_ROOT_DEDUP=0`.
- Extended the leaf-kernel audit with shell-query and root-dedup counters.
- Added `validation/check_h2_shell_root_pruning_model.py` plus Rust shell-pruner unit tests.
- H0 is unchanged.

## v16 experimental — plateau-native leaf zero-death elision

- Hierarchical H0/H2 leaves now discard finite zero-persistence local deaths directly at the union decision, before constructing local merge records or entering the leaf history contractor.
- The optimization preserves sequential voxel activation and the exact neighborhood-pruning invariant; it does **not** pre-activate an entire equal-value plateau. Interface merges, boundary-state-changing attaches, and H2 Outside transitions are unchanged.
- Added `BETTI_HIER_PLATEAU_NATIVE_LEAF=0` for same-binary v15 A/B validation and profiling.
- Added `leaf_plateau_zero_local_merges_elided` profiling plus `validation/check_plateau_native_leaf_elision_model.py` and `docs/plateau-native-leaf-elision.md`.

## v15 experimental — watched zero-history leaf fast path

- Leaf-local finalized history now stays in compact slab-local IDs until export, avoiding full branch construction for the tens of millions of zero-persistence events that disappear inside a leaf.
- Added an exact one-bit-per-local-branch watch filter. When a finalized child is not referenced by any retained positive event, the local contractor skips the sparse parent `HashMap` lookup entirely; filtration-aware repair is unchanged for watched parents.
- Added H0/H2 profiling counters for watch checks, skipped lookups, actual parent-map lookups, and zero-persistence fast drops. `BETTI_HIER_LEAF_HISTORY_WATCH_FILTER=0` disables only the watch shortcut for A/B profiling.
- Added `validation/check_leaf_history_watch_filter_model.py` and the design note `docs/leaf-history-watch-filter.md`.

## v14.1 experimental — compact fused leaves compile fix

- Fixed the H0/H2 compact `LocalUnionFind::new` constructors to bind the slab-local `voxel_count` from `Block::voxel_count()` before allocating packed state.
- Added a debug assertion that `block.values.len()` matches the slab voxel count.
- No algorithmic or profiling semantics changed from v14.


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

- Added an opt-in v17 leaf-kernel audit (`BETTI_HIER_LEAF_KERNEL_AUDIT=1`). It preserves the v16 reducer while counting active-neighbor pruning, representative edges, successful/no-op union attempts, union classes, carried-root reuse, H2 Outside involvement, and union-find parent depths. Timing is sampled at a configurable stride (`BETTI_HIER_LEAF_AUDIT_SAMPLE_STRIDE`, default 4096) to keep diagnostic overhead bounded; the external branch-tree profiler records both raw counters and derived ratios.

- Experimental compact branch-tree leaf state replaces the per-voxel full H0/H2 branch object, separate rank array, and separate active byte with packed parent/rank state plus one local elder ID and one interface representative. Branch birth/global IDs are reconstructed from the slab values only when an event is emitted. The leaf sweep also uses the existing interior linear-offset neighborhood kernel and carries the current UF root across representative unions. This is intended to make d16/d32 leaves practical without giving up four-way leaf concurrency, reducing hierarchy seams and interface replay at the source.
- The concurrent-leaf memory estimator is reduced from 128 to 48 bytes/voxel for this compact layout; `BETTI_HIER_LEAF_WORKERS` and `BETTI_HIER_LEAF_BUDGET_MB` remain explicit overrides. Profiling now reports the compact leaf-state capacity and layout.

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

## Unreleased - U16 H0 persistence parallel leaf preparation

- Prepare native-U16 hierarchical H0 persistence leaves in bounded Rayon batches.
- Buffer only positive finalized leaf pairs per worker, then hand them off in slab order before deterministic hierarchical fan-in.
- Add `BETTI_PERSIST_H0_LEAF_WORKERS` and `BETTI_PERSIST_H0_LEAF_BUDGET_MB` controls.
- Add exact worker-count interval-equivalence validation and a dedicated worker/depth profiler.
