# H2 phase-by-phase memory audit (v1.11)

v1.10 found that the packed local union-find was faster for H2 but produced an unexpected ~50 MiB increase in external peak RSS on CX09T1. v1.11 deliberately makes no new topology optimization. It adds a Linux-only diagnostic flag:

```text
--h2-memory-audit
```

The flag samples `/proc/self/status` and `/proc/self/smaps_rollup` at major H2 streaming stages and prints `PROFILE_MEM` records containing live RSS, high-water RSS, anonymous/file-backed RSS and PSS, and private/anonymous mappings. It also prints `PROFILE_MEMVEC` records with actual vector lengths, capacities, element footprints, UF parent/rank storage, local birth state, event buffers, boundary faces, retained interface faces, reduction state, and reader buffers.

The audit is read-only: it does not change persistence ordering, pruning, union policy, or storage layout. It is intended to be run separately from ordinary timing benchmarks because reading `smaps_rollup` adds diagnostic overhead.

Recommended CX09T1 comparison:

```bash
python3 scripts/profile_scalar_h2_memory_audit.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 3 \
  --output cx09t1_h2_memory_audit_d16.csv
```

The script requires exact persistence hashes across `parent-rank` and `packed`. The main diagnostic question is the first stage at which packed mode's `hwm_kb` or live `rss_kb` diverges from parent-rank. Compare that divergence with the matching `PROFILE_MEMVEC` capacity rows:

- larger declared vector capacity => an explicit allocation difference;
- equal/smaller capacities but higher anonymous RSS => allocator/page-residency effect;
- divergence only after local processing has returned => retained event/face state or allocator retention;
- divergence only during reduction => the local packed layout is not itself the peak source.
