
## v1.24.1 candidate harness guard

- Added a validator preflight that rejects stale binaries before staging the F32 fixture.
- Added parser regression tests for both hierarchical scalar stream modes.
- Updated the invalid-mode diagnostic to list `h2-scalar-hierarchical-stream`.

# v1.24 candidate — outside-aware hierarchical H2

- Adds experimental `h2-scalar-hierarchical-stream` for exact outside-aware online H2 fan-in.
- Keeps F32 keys native at 4 bytes through leaf summaries, disk runs, pairwise hierarchy, and compact final reduction.
- Reuses the decoded F32 leaf buffer as compact H2 birth state, keeps hierarchical boundary IDs implicit, and emits leaf events directly to disk.
- Applies exact superlevel elder-dominated attach finalization in leaves and fan-ins.
- Uses packed parent/rank plus 4-byte F32 births for an 8-byte/node pairwise H2 state bounded by at most four face areas.
- Preserves the deterministic H2 event priority `outside -> attach -> interface -> cross`.
- Uses a terminal-free final fan-in with one structural distinguished outside node and zero materialized root-summary files.
- Adds exact CX09T1/F32 equivalence and paired flat-vs-hierarchical profiling tools.
- The outside-aware summary algebra passed 13,600 independent randomized small-volume cases before packaging; Rust compilation and CX09T1 remain required gates.

# v1.23 candidate

- Adds exact `elder-dominated` pruning for disk-backed hierarchical H0 attach events.
- An attach whose branch birth is no older than the terminal component birth is finalized immediately rather than propagated to higher fan-in levels.
- Adds combine-level attach byte/count instrumentation, CX09T1 equivalence/profile tools, and real-prefix profiler support for the pruning strategy.

# v1.22 candidate — terminal-free hierarchical H0 root

- v1.21 full-CX09T1 profile preserved the exact H0 barcode at d8/d16/d32 and kept the optimized pair frontier fixed at 4A.
- The optimized hierarchy reduced median RSS to 15.8 MiB at d8, but the materialized root summary still wrote about 253--313 MiB of attach events on CX09T1.
- The final fan-in is now terminal-free: it reduces the last two child summaries directly with the ordinary packed persistence UF and emits remaining finite pairs plus essential births.
- Multi-slab hierarchical runs no longer create or reread root attach/interface files; profiling reports `root_materialized=false`, zero root-summary bytes, and `final_interface_nodes=0`.
- Intermediate pairwise summaries, direct/reuse F32 leaves, exact event ordering, and the <=4A frontier are unchanged.
- Adds a v1.22 profile harness and tightens the hierarchical equivalence gate to require zero root materialization.

# v1.21 candidate — optimized disk-backed hierarchical H0

- v1.20 architecture equivalence gate passed on CX09T1.
- v1.20 profile confirmed the hierarchical frontier is exactly 4A across d8/d16/d32 while final state is 2A, independent of slab count.
- F32 H0 direct+reuse reduced staged-fixture RSS about 50.1% and median wall about 4.8%; H2 native32 reduced RSS about 42.3%, temporary runs about 37.0%, and median wall about 26.8%.
- Adds `h0-scalar-hierarchical-stream`: native-F32 direct/reuse leaves, disk-backed live summaries, online binary fan-in, immediate child-run deletion, and packed pairwise UF with terminal-root invariance.
- Pairwise H0 composition has explicit state 8 bytes/node and a measured/theoretical frontier bound of at most 4 face areas.
- Adds exact CX09T1/F32 validation and a paired full-CX09T1 profiler.
- `--merge-strategy auto` remains experimental: the v1.20 32-reader heuristic was not supported by profiling, and production/reference merge remains `scan`.

# v1.20.0 architecture candidate

- Experimental H2 native-F32 end-to-end storage through slab summaries, temporary runs, interface births, and compact global reduction.
- Experimental H0 `reuse-input` birth storage and direct event sinks for bounded large-slab event memory.
- Experimental exact `h0-scalar-hierarchical` online binary fan-in mode.
- Experimental `--merge-strategy auto` with auditable reader-count selection.
- Added CX09T1/F32 exact-equivalence validators and an aggregate architectural candidate gate.
- v1.19 production defaults remain the reference until these candidates pass real-data equivalence and profiling.

# v1.19 production — packed global H0 + promoted end-to-end native F32


- Promotes the v1.18 end-to-end native-F32 H0 storage path after exact real-F32 validation: identical 374,303-interval canonical hash, median wall time 23.433 -> 19.949 s (-14.9%), peak RSS about 1013 -> 688 MiB (-32.1%), temporary-run bytes -30.0%, explicit global H0 state -30.77%.
- Promotes `--global-h0-uf-layout packed` after the balanced five-pair real-F32 profile: identical persistence, zero separate rank bytes, explicit global state 246.84 -> 219.41 MiB on the two-slice test, median global reduction time about -6.0%, and no material end-to-end regression.
- Production H0 now uses `f32_key_mode=native32` end-to-end for F32 input and `global_h0_uf_layout=packed`; `legacy64` and `parent-rank` remain reference/debug options.
- For the 3792 x 3792 x 2048 F32 target at slab depth 32, the deterministic explicit global H0 state estimate is about 13.71 GiB instead of 22.28 GiB in v1.17.
- See `H0_F32_END_TO_END_INVESTIGATION.md` and `GLOBAL_H0_UF_LAYOUT_INVESTIGATION.md`.

# v1.19 candidate — packed global H0 union-find

- Adds `--global-h0-uf-layout parent-rank|packed`; `parent-rank` remains the candidate-tree default until profiling.
- The packed layout stores root rank in the `u32` parent word and removes the separate rank byte.
- On native-F32 H0 this changes explicit global state from 9 to 8 bytes/interface node (11.11%); at d32 on the 3792 x 3792 x 2048 target this is about 15.43 -> 13.71 GiB.
- Adds `PROFILE_GLOBAL_H0_STATE`, an independent equivalence validator, and a balanced real-F32 layout profiler.
- Builds on the successful v1.18 two-slice real-F32 A/B result: identical 374,303-interval hash, median wall -14.9%, peak RSS -32.1%, temporary-run bytes -30.0%, explicit global H0 state -30.77%.
- See `GLOBAL_H0_UF_LAYOUT_INVESTIGATION.md`.

# v1.18 candidate — end-to-end native F32 H0 storage

- Extends `--f32-key-mode native32` through H0 slab summaries, cross-interface sparsification, temporary disk runs, and global H0 birth state.
- Keeps `legacy64` as the same-binary reference path and leaves H2 unchanged.
- Reduces explicit global H0 state on F32 input from 13 to 9 bytes/interface node (30.77%).
- Adds `PROFILE_H0_STORAGE` accounting for disk-key width, temporary-run bytes, and estimated global H0 state.
- Adds a guarded real-F32 end-to-end validator and extends the F32 profiler with disk/global storage metrics and optional slice-subset staging.
- See `H0_F32_END_TO_END_INVESTIGATION.md`.

# v1.17 production consolidation

- Promotes `--local-h2-birth-state compact` to the H2 production default after the guarded CX09T1 equivalence/profile campaign.
- Retains `tagged` as the explicit local-H2 reference/debug backend.
- Keeps global H2 `parent-rank` as the frozen production layout; packed global rank encoding remains an ablation candidate.
- Updates the production equivalence gate and full profiler to require the compact local-H2 default.
- Makes the full production profiler default to slab depths 8, 16, and 32 and records/validates the scalar source pixel type.
- Corrects release documentation: CX09T1 is reported by the scalar loader as U16; `native32` is relevant only for F32 stacks.
- Freezes the no-tuning-flag production configuration in `PRODUCTION_RELEASE.md`.
- Generalizes the release-gate environment names to `SCALAR_TEST_STACK` / `SCALAR_TEST_CONNECTIVITY` while retaining the older F32-specific names as compatibility fallbacks.

# Changelog

## v1.15.1

- Adds stale-release-binary preflight checks to the global H2 UF equivalence validator and profiler. No Rust/runtime algorithm changes.

# v1.15 — global H2 union-find layout ablation

- Promotes `--global-h2-birth-state compact` after CX09T1 exact-equivalence and profiling: explicit global H2 state 155,557,477 -> 54,655,333 bytes, median peak RSS -6.5%, reduction time -7.1%, wall time -1.8%.
- Corrects the historical tagged allocation estimate: the 16-byte enum vector doubled capacity when the outside node was appended, yielding about 32 allocated birth bytes/node and ~37 total global-UF bytes/node in the measured run.
- Adds `--global-h2-uf-layout parent-rank|packed` for compact global H2 reconciliation.
- Packed global roots reserve only the top 256 `u32` words for rank markers, preserving almost the full 32-bit global node range while eliminating the separate rank byte.
- Adds exact parent-rank-vs-packed unit/end-to-end validation and balanced wall/RSS profiling.
- Keeps global `parent-rank` as the v1.15 reference default pending real-data profiling.

# v1.14 — global H2 birth-state ablation

- Added `--global-h2-birth-state tagged|compact` for scalar H2 streaming reconciliation.
- Added a compact outside-root representation that reuses the 8-byte `ScalarKey` birth vector directly instead of converting it to the tagged `BackgroundBirth` enum.
- Added exact reference-vs-compact unit and end-to-end validation plus external wall/RSS profiling.
- Added `PROFILE_GLOBAL_H2_STATE` with exact parent/rank/birth capacities.
- `tagged` was retained as the v1.14 reference default; v1.15 promotes `compact` after successful real-data ablation.

# v1.13.1 production Clippy hotfix

- Fixes the production release gate under `cargo clippy --all-targets -- -D warnings`.
- Removes stale `active_state` fields from local H0/H2 persistence union-find wrappers.
- Compiles native-F32 compatibility helpers and the callback-style interior-pruning helper only for tests when they are not used by the production binary.
- Removes the duplicated `unsafe_code` allowance from `allocator_trim`; the sole unsafe exception remains the module-level allowance on `mod allocator_trim` in `main.rs`.
- Rewrites the interior-neighbor indexing loop using `enumerate()` and replaces a fixed-size test `Vec` with an array.
- Adds narrow, documented `clippy::too_many_arguments` allowances only on internal ablation kernels whose explicit knobs are intentionally retained for reproducibility.
- No production algorithm, default, persistence semantics, or phase-trim behavior changed.

# v1.13 production consolidation

- Promotes the measured scalar-stream production defaults from the CX09T1
  ablation campaign while retaining every reference/legacy path as an explicit
  command-line option.
- Uses `packed` local union-find for H0 and H2, `separate` active state for H0,
  `parent-sentinel` active state for H2, and a one-time H2 `before-reduce`
  `malloc_trim(0)` on Linux/glibc.
- Applies H0/H2 defaults per mode rather than through one global tuning struct;
  explicit options override only the setting the user supplied.
- Adds `validation/check_production_scalar_equivalence.py`, which verifies the
  no-ablation production defaults against the independent in-memory scalar
  persistence implementation at multiple slab depths.
- Adds `scripts/check_production_release.sh` as the consolidated release gate:
  Rust tests, Clippy, Python tests, syntax checks, release build, black-box
  reference checks, and optional real-stack production equivalence.
- Adds `scripts/profile_production_scalar_stream.py` for multi-specimen,
  multi-depth H0/H2 profiling. The profiler records binary SHA-256, effective
  `PROFILE_CONFIG`, phase timings, peak RSS, and exact persistence hashes, and
  rejects any barcode change across repeats or slab depths.
- Extends H0 `PROFILE_CONFIG` output to include the effective local UF layout.

# v1.12 H2 phase-boundary trim ablation

- Added `--phase-trim <off|before-reduce>` for exact scalar H2 streaming.
- `before-reduce` performs one glibc `malloc_trim(0)` after slab preparation and immediately before global reduction.
- Added `PROFILE_TRIM` timing/result output.
- Added exact in-memory/stream equivalence validation across both UF layouts and both trim modes.
- Added a balanced external wall/RSS profiler for the four H2 configurations.
- Kept phase trimming off by default until the CX09T1 ablation is measured.
- Narrowed the unsafe-code exception to `allocator_trim`; all topology/UF code remains under `deny(unsafe_code)`.

# Changelog

## 2026-09-03 — v1.11 H2 memory-stage audit

- Adds `--h2-memory-audit`, a read-only Linux `/proc` diagnostic for scalar H2 streaming.
- Emits stage-resolved live RSS and high-water RSS plus anonymous/file-backed RSS/PSS from `smaps_rollup`.
- Emits actual vector length/capacity footprints for slab values/order, local UF parent/rank, local birth state, active/interface state, pruning cache, event buffers, boundary faces, retained faces, and global reduction state.
- Adds `scripts/profile_scalar_h2_memory_audit.py` to compare `parent-rank` and `packed` in fresh processes, require exact persistence hashes, and identify the stage with the largest matched HWM divergence.
- Makes no new topology or allocation optimization; v1.10 computation remains unchanged when the audit flag is absent.

## 2026-09-03 — v1.9 interface-state investigation

- Adds `--interface-state vector|root-invariant` for local scalar persistence.
- `root-invariant` allocates no per-voxel `interface_rep` vector. A component that touches a slab z-interface is forced to retain an interface voxel as its union-find root.
- Rank remains authoritative when both or neither merging components touch an interface; one-interface merges force the interface root and update its stored rank as a height bound.
- Adds diagnostics for interface-representative queries/writes, forced interface-root unions, interface-interface unions, maximum observed rank, and the local interface-state allocation.
- Adds exact in-memory/vector/root-invariant persistence checks plus balanced wall/RSS and operation-count profilers.
- Keeps `vector` as the default until the real-stack ablation is measured.

## 2026-09-03 — v1.4 local-sweep investigation

- Promotes the measured `interior-fast` scalar persistence neighbor kernel to the default.
- Adds `--representative-active-check recheck|trust-pruner` to ablate the redundant post-pruner active load.
- Adds `--sweep-diagnostics` and `PROFILE_SWEEP` operation counters for pruning/cache/union-find analysis.
- Adds exact in-memory/recheck/trust-pruner persistence validation and balanced A/B profiling utilities.
- Keeps `recheck` as the default until the new ablation is measured.

## 2026-09-03 — v1.3 preparation profiler and neighbor-kernel ablation

- Added detailed scalar-stream preparation timings for TIFF decode, key conversion, slab copy, scalar ordering, local topology sweep, local run writing, and cross-interface work.
- Added `--neighbor-kernel <generic|interior-fast>`; `generic` remains the default until measured.
- Added an interior-voxel neighbor path using precomputed linear offsets while retaining the original boundary-safe path.
- Added exact in-memory/generic/interior-fast persistence equivalence validation.
- Added a balanced same-binary neighbor-kernel profiler with persistence hashes, phase timings, RSS/I/O metrics, and fast-path voxel coverage.

## Unreleased revision

### Mathematical and interface corrections

- Centralized the fixed sublevel/strict-background, dual-connectivity, virtual
  outside, and half-open interval contract in the documentation.
- Renamed the preferred tree commands to `branch-tree-*`; retained
  `merge-tree-*` as compatibility aliases and documented equal-value plateau
  ambiguity.
- Preserved source numeric types, normalized signed zero exactly, rejected
  nonfinite scalar input, and added exact counting/radix ordering paths.
- Added checked dimension and identifier planning through `--dry-run-plan`.
- Unknown command-line options now fail instead of becoming accidental paths.
- The `--` terminator now protects even literal help- or version-like paths.

### Scalar-stream optimization audit

- Added same-binary runtime ablation switches for run merging (`scan|heap`),
  scalar interface ordering (`comparison|radix`), and slab-event ordering
  (`resort|verify`). CX09T1 factorial profiling selected `scan + radix + verify`
  as the production default for both H0 and H2.
- Added `--f32-key-mode <legacy64|native32>` for disk-backed scalar persistence.
  F32 slabs can now retain exact four-byte ordered keys locally, use four radix
  passes instead of eight, and keep local persistence birth state at native
  width. Boundary/run records remain in the canonical 64-bit ScalarKey format,
  so the global reducer and CSV representation are unchanged.
- Added same-binary native32-vs-legacy64 correctness and profiling tools.
- Added a full-factorial ablation profiler that measures all eight combinations
  and rejects any configuration that changes the canonical persistence
  multiset.
- Changed the A/B profiler to warm the workload by default and alternate
  baseline/candidate order between repeats to reduce cache/order bias.
- Extended the representative stream/in-memory equivalence checker so any
  ablation combination can be checked against the independent in-memory path.
- Replaced scalar H0/H2 persistence run-wide minimum/maximum scans with
  priority-queue merges. Run selection is now logarithmic in the number of
  active run files instead of rescanning every run at each distinct value.
- Replaced scalar cross-face comparison sorting with the same stable exact
  radix-order backend used by floating slab sweeps. Tie order remains
  value-then-index for sublevel processing and reverses exactly for the
  superlevel path.
- Removed redundant per-slab persistence-event copies and comparison sorts.
  H0/H2 run writers now verify the monotonicity guaranteed by the local sweep
  while serializing events directly from the summary vectors.
- Added machine-readable phase timing lines for scalar H0/H2 streaming
  persistence: slab preparation, global reduction, temporary-run cleanup, and
  total time.
- Added an A/B profiler that records wall time, peak RSS, Linux I/O/page-fault
  counters, internal phase timings, and an exact canonical hash of the
  persistence multiset.
- Added a representative-stack stream/in-memory equivalence checker and a
  one-command regression wrapper around the existing independent validation
  campaign.

### Performance and memory

- Replaced materialized 26-connected cross-face candidate arrays by an exact
  vertex-activation Kruskal sweep with `O(A)` state.
- Kept a direct path for 6-connected matching faces.
- Computed slab boundary identifiers instead of retaining full optional
  metadata arrays.
- Allocated large local-pruning caches only for 26-connectivity.
- Reworked the Betti-curve distance matrix to keep at most two curve inputs
  open and one flat upper triangle of doubles.

### Reliability

- Added transactional final-output writers in Rust and Python.
- Added a real Cargo dependency lockfile and a pinned Rust toolchain.
- Fixed a background event bug in which the “no interface representative”
  sentinel could be emitted and then indexed as a real node.
- Added independent graph-barcode checks, branch invariants, exhaustive tiny
  cubical boundary-matrix checks, focused TIFF validation tests, and broader
  black-box mode comparisons.

### Documentation

- Added architecture, contribution, validation, and release-packaging guides.
- Replaced broad “streaming” language with separate memory, disk, and global
  topology bounds.
- Added a primary-source comparison with the June 2026 Flash Cubical preprint,
  including an explicit claim boundary and matched diagnostic checks.
- Renamed the local helper type from `LocalPruningLookup` to
  `NeighborhoodComponentPruner` so its on-demand graph role is not confused
  with a precomputed cubical lower-star table.

### Deliberately deferred

- A proof-backed composable two-face summary that removes global `O(G)` state.
- Bounded-fan-in external merges with manifests, checksums, and restart.
- Canonical plateau merge-tree nodes.
- Direct-to-run local persistence event sinks that avoid retaining slab event vectors.
- Error-bounded quantization, a general volume source, and full provenance.

### v1.5 root-carrying union investigation

- Added `--union-kernel <conventional|root-carrying>` for scalar H0/H2 persistence streams.
- Root-carrying local unions preserve the existing persistence merge/action logic but carry the surviving current-component root across one voxel's representative-neighbor loop, avoiding one `find()` per union attempt.
- Extended `PROFILE_SWEEP` with `root_carry_attempts` and `avoided_find_calls`.
- Added exact in-memory/conventional/root-carrying equivalence validation, balanced timing, and operation-count comparison scripts.
- Kept `conventional` as the default until representative profiling demonstrates a wall-clock win.

### v1.6 H2 direct-parent root shortcut investigation

- Promoted `root-carrying` to the scalar-stream union default after CX09T1 v1.5 profiling showed it was neutral for H0 and consistently faster for H2.
- Added `--neighbor-root-check <find|parent-shortcut>` for the H2 root-carrying local union path. `find` remains the v1.6 reference/default; `parent-shortcut` first accepts `neighbor == current_root` or `parent[neighbor] == current_root` as a proven same-component case and otherwise falls back to the existing `find(neighbor)` path.
- Extended H2 `PROFILE_SWEEP` counters with `direct_parent_checks`, `direct_parent_hits`, and `avoided_neighbor_find_calls`.
- Added exact in-memory/find/parent-shortcut equivalence validation, balanced wall-time profiling, and diagnostic operation-count comparison scripts.

## v1.7 H0 pruning-cache ablation

- Promotes the measured H2 `parent-shortcut` root check to the normal default.
- Adds `--h0-pruning-cache off|4k|16k|64k|256k`; historical `64k` remains the
  default pending measurement.
- `off` allocates no 26-neighbor pruning cache; enabled modes allocate a bounded
  direct-mapped cache of the requested power-of-two size.
- Separates cache lookups from component-mask computations in sweep diagnostics.
- Adds exact in-memory-vs-stream cache equivalence validation and balanced
  wall/RSS plus diagnostic cache profilers.

## v1.8 active-state investigation

- Added `--active-state separate|parent-sentinel` for exact scalar H0/H2 persistence streams.
- `parent-sentinel` reserves `u32::MAX` as inactive and eliminates the slab-local byte active array.
- Added representative-list pruning APIs so parent state can be read before mutable UF unions without `unsafe` aliasing.
- Added `active_state_checks` to local-sweep diagnostics.
- Added exact active-state equivalence validation plus balanced timing/RSS and diagnostic profilers.
- Historical `separate` remains the default until the real-data ablation is measured.

## v1.10 - local union-find layout ablation

- Added `--uf-layout parent-rank|packed` for scalar H0/H2 local persistence.
- Added a tagged packed-u32 local DSU representation that removes the
  byte-per-voxel rank vector while retaining rank-based union and the
  root-invariant interface policy.
- Packed mode explicitly supports up to 2^31 voxels per slab; parent-rank
  retains the previous u32-index capacity.
- Added UF parent/rank state byte diagnostics and exact parent-rank/packed
  persistence regression tests.
- Added balanced timing/RSS and operation-count profiling scripts plus an
  independent in-memory persistence equivalence checker.
- Promoted `root-invariant` interface state to the default after the v1.9
  CX09T1 measurements showed exact persistence with lower runtime for H0/H2.

## v1.16 - local H2 birth-state ablation

- Adds `--local-h2-birth-state tagged|compact`.
- Keeps `tagged` as the reference default pending real-data profiling.
- Compact storage encodes the outside state with a reserved non-finite scalar-key marker and
  stores finite births at native key width.
- Adds exact in-memory-oracle equivalence validation and balanced external wall/RSS profiling.
- Emits `PROFILE_LOCAL_H2_STATE` for the first slab so explicit local UF allocation is visible
  without enabling the intrusive H2 memory audit.

## v1.16.2 - native-F32 local-H2 benchmark guard

- Hardened the local H2 birth-state equivalence checker and A/B profiler to pass
  `--f32-key-mode native32` explicitly.
- The harnesses now parse the scalar source pixel type and `PROFILE_CONFIG`; an F32
  run fails unless the effective key mode is `native32` and
  `PROFILE_LOCAL_H2_STATE` reports `key_bytes=4`.
- The profiler records `source_pixel_type` and `reported_f32_key_mode` in its CSV so
  performance results retain the representation provenance needed for promotion.


## v1.23.1 profiling-harness correction

- Fixes `profile_v1_23_attach_pruning.py` so pruning exactness is judged from the persistence multiset, not CSV emission order.
- Records both `output_sha256_ordered` (diagnostic) and `output_sha256_canonical` (exactness gate).
- On a genuine mismatch, reports the first missing/extra `(birth, death)` intervals.
- No Rust implementation changes.

## v1.24.2 H2 profiler elapsed-time parser fix

- Fixes `scripts/profile_v1_24_h2_hierarchical_stream.py` parsing of GNU
  `/usr/bin/time -v` elapsed-wall-clock output.  The previous lazy regular
  expression could stop at the `h:mm:ss` text inside the label and pass a
  non-numeric suffix to `wall_seconds()`.
- The parser now anchors to the complete elapsed-time line and accepts plain
  seconds, `m:ss[.ff]`, and `h:mm:ss[.ff]` values.
- Adds an explicit diagnostic when an elapsed-time line is present but cannot
  be parsed.
- No Rust or persistence-algorithm changes; no rebuild is required when
  updating from v1.24.1.

## v1.25 H2 direct cross-interface candidate

- Added `--h2-hier-cross-storage <disk|direct>` for hierarchical scalar H2.
- `direct` preserves H2 event priority while feeding the sparse cross-interface
  forest directly into pairwise reconciliation, eliminating the temporary
  cross-run write/read cycle.
- Added exact disk-vs-direct-vs-in-memory validation and a filesystem-I/O-aware
  CX09T1 profiler.

## v1.25.1 — H2 cross-profile compatibility fix

- Fixed the v1.25 direct-cross validator to recognize legacy untagged hierarchical H2 combine/final profile records as the disk cross-storage path.
- Direct cross-storage records must still be explicitly tagged `cross_storage=direct`.
- Hardened the v1.25 H2 cross-storage profiler to parse disk/direct per-line cross records consistently.
- No Rust or persistence-algorithm changes; rebuilding `betti_curves` is not required.

## v1.26 candidate

- Promotes direct cross-interface consumption as the hierarchical H2 default after exact CX09T1 v1.25 profiling.
- Adds `--h2-hier-outside-structural-pruning <off|outside-dominated>`.
- Adds exact outside-aware canonicalization of terminal structural summary events in hierarchical H2 leaves and fan-ins.
- Adds `outside_structural_elided` profiling and leaf/parent summary-byte profiling harnesses.
- Adds a strict flat/off/pruned exactness gate on an exact-F32 fixture.

## v1.26.1 real-F32 H2 prefix profiling harness

- Added `scripts/profile_large_h2_hierarchical_prefix.py`.
- The harness stages real TIFF prefixes with symlinks, monitors peak scratch use,
  records GNU-time CPU/RSS/filesystem counters, and aggregates hierarchical H2
  leaf/parent summary bytes.
- Defaults to the current optimized H2 candidate: native F32 keys, compact local
  and global H2 births, packed global UF, direct cross consumption, and
  outside-dominated structural pruning.
- No Rust implementation changes relative to v1.26.0.
