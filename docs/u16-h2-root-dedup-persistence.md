# U16 H2 persistence exact UF-root deduplication (v23 experimental)

## Motivation

The v22 native-U16 H2 persistence profile showed that plateau-zero pair elision is nearly neutral at the production-relevant slab depth 16. The branch-tree v18 audit had previously shown a different source of H2 redundancy: many active-neighbor edges connect the newly activated voxel to neighbors that already belong to the same union-find component.

v23 tests whether the same exact reduction transfers to the native-U16 hierarchical H2 persistence leaf kernel.

## Transformation

For one newly activated voxel, let the active representative neighbors occur in the deterministic existing order

`n_1, n_2, ..., n_m`.

Before inserting any center edge, v23 resolves each neighbor to its current UF root. It retains the first occurrence of each distinct root and then performs persistence unions in that retained order.

The implementation uses a fixed `[u32; 26]` stack buffer. There is no hash table or heap allocation in the dedup step.

## Exactness argument

If two representatives have the same pre-existing UF root, they are already connected before either center edge is inserted. After the first edge from the new voxel to that component is processed, any later edge to another representative of the same component is a same-root union and cannot change:

- the component partition,
- the component birth state,
- Outside membership,
- the retained interface representative,
- or the emitted persistence action sequence.

Keeping the first occurrence of each root preserves the order in which distinct components are merged. Therefore root dedup removes only edges that are provably no-ops in the reference execution.

## Controls

The experiment is **disabled by default** for native-U16 hierarchical H2 persistence after the paired v23.1 benchmark showed a reproducible slowdown.

```bash
BETTI_PERSIST_H2_ROOT_DEDUP=0   # production reference/default
BETTI_PERSIST_H2_ROOT_DEDUP=1   # rejected v23 experimental ablation
```

`BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION=1` is held fixed during the v23 A/B benchmark.

## Profiling

The U16 H2 leaf profile now reports exact lightweight hot-path counters even when the expensive general sweep diagnostics are disabled:

- `union_attempts`
- `successful_unions`
- `same_root_unions`
- `root_dedup_inputs`
- `root_dedup_unique`
- `root_dedup_skipped`

The dedicated benchmark is `scripts/profile_u16_h2_root_dedup.py`. The v23 follow-up uses a paired, interleaved design rather than running all OFF measurements before all ON measurements. For each slab depth, odd pairs run OFF→ON and even pairs run ON→OFF (or the reverse when `--start-with on` is requested). This removes systematic run-order drift from the treatment comparison.

The default confirmation profile is d16/d32/d64 with one paired warmup and 11 recorded pairs per depth. It writes raw runs, per-backend robust summaries, paired rows, and a paired summary. The paired summary reports median percentage deltas together with MAD and IQR, the fraction of pairs where dedup is faster, an exact two-sided sign test, and the difference between OFF-first and ON-first median wall-time effects as a residual run-order diagnostic; the raw backend summary also reports wall-time coefficient of variation. Every accepted pair checks that successful-union and zero-persistence-elision counts agree, that the enabled run has no same-root union attempts, and that the skipped-root count exactly accounts for the reference same-root unions.

Use the end-to-end candidate gate:

```bash
scripts/check_u16_h2_root_dedup_candidate.sh /path/to/CX09T1
```

## Validation

- `validation/check_u16_h2_root_dedup_persistence_model.py` checks the first-distinct-root transformation on randomized component metadata/orderings.
- `validation/check_u16_h2_root_dedup_persistence_equivalence.py` now checks d16/d32/d64 by default. For each depth it requires identical ordered persistence rows for root dedup off/on, verifies the mode reported by the binary, and also checks interval-multiset invariance across slab depths. `--allow-order-difference` weakens only the within-depth row-order requirement when investigating a harmless ordering change.

## v23.1 decision

The interleaved CX09T1 profile rejected root dedup as a production optimization. Median paired wall-time changes for dedup ON were approximately +2.0% at d16, +4.0% at d32, and +6.2% at d64; the local sweep showed the same direction. Dedup still removed about 66% of nominal union attempts, demonstrating that nominal union-attempt count is not a valid performance proxy for this kernel.

The likely reason is that the production root-carrying path already rejects many same-component encounters through the much cheaper direct-parent shortcut, whereas eager root dedup pays for `find()` on every representative. The implementation and validators are retained as an exact negative ablation, but `BETTI_PERSIST_H2_ROOT_DEDUP` is now opt-in.

The paired raw data are retained under `benchmarks/data/v23_u16_h2_root_dedup_paired*.csv`. The earlier grouped d16/d32 timing remains exploratory only. The follow-up candidate is the narrower two-hop parent shortcut described in `docs/u16-h2-two-hop-parent-shortcut.md`.
