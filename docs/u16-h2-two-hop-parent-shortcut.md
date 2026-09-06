# U16 H2 two-hop parent shortcut (v23.2 / v23.2.1 experimental)

## Motivation

The paired v23.1 experiment rejected full representative-root deduplication for native-U16 hierarchical H2 persistence. Although it removed about 66% of nominal union attempts, it made the local sweep slower because every representative paid for an eager `find()` plus distinct-root scanning. The existing production one-hop parent shortcut wins by rejecting many same-component encounters with only a cheap parent-array test.

v23.2 therefore tests a narrower idea: after the existing one-hop shortcut misses, inspect exactly one additional parent link before falling back to `find(neighbor)`.

## Candidate transformation

Reference (`parent-shortcut`):

```text
neighbor == current_root
or parent[neighbor] == current_root
otherwise find(neighbor)
```

Candidate (`parent-two-hop`):

```text
neighbor == current_root
or parent[neighbor] == current_root
or parent[parent[neighbor]] == current_root
otherwise find(neighbor)
```

The grandparent read is attempted only when the direct-parent test misses. It is read-only and does not perform path compression. Packed union-find roots carry a tagged rank word in their own slot, so the implementation first checks whether the intermediate parent is already a root before indexing through it.

## Exactness

If `parent[parent[neighbor]] == current_root`, then `neighbor` is already in the component represented by `current_root`. The reference fallback `find(neighbor)` would therefore return `current_root`, after which the union is a same-root no-op. Returning immediately preserves the component partition, persistence birth state, Outside state, interface representative, and event sequence.

The optimization changes only how an already-proven same-component case is recognized. It does not reorder distinct-component unions.

## Controls

The production reference remains the default:

```text
--neighbor-root-check parent-shortcut
```

The candidate is opt-in:

```text
--neighbor-root-check parent-two-hop
```

The rejected native-U16 root-dedup experiment is now opt-in only:

```text
BETTI_PERSIST_H2_ROOT_DEDUP=0   # default/reference
BETTI_PERSIST_H2_ROOT_DEDUP=1   # rejected v23 ablation only
```

## v23.2.1 profiling correction

The v23.2 paired timing runs deliberately left `--sweep-diagnostics` disabled, which was correct for uncontaminated timing but meant the new diagnostic columns were all zero. Those zeros do **not** mean the two-hop branch was skipped. v23.2.1 separates the two concerns:

- paired A/B timing runs: diagnostics OFF;
- structural diagnostic runs: `--sweep-diagnostics` ON, never used for speed claims.

Existing v23.2 timing measurements therefore remain usable. Only the diagnostic-counter interpretation needs a diagnostic-only rerun.

## Instrumentation

The native-U16 H2 leaf profile reports:

- `direct_parent_checks`, `direct_parent_hits`
- `two_hop_parent_checks`, `two_hop_parent_hits`
- `avoided_neighbor_find_calls`
- `neighbor_find_calls`, `neighbor_find_parent_steps`
- fallback-find path-depth buckets: 0, 1, 2, and >2 parent hops
- the same path-depth buckets restricted to fallback finds that still resolve to `current_root`

These counters answer the key question before a production freeze: how many expensive same-root fallback finds actually have depth two, and how many remain deeper than two?

## Validation and paired profiling

Exact multi-depth equivalence:

```bash
python3 validation/check_u16_h2_two_hop_parent_shortcut_equivalence.py \
  /path/to/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26
```

Paired benchmark:

```bash
python3 scripts/profile_u16_h2_two_hop_parent_shortcut.py \
  /path/to/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26 \
  --warmup 1 \
  --repeats 11 \
  --start-with reference \
  --output profiles/v23_2_u16_h2_two_hop_parent_shortcut_paired.csv
```

The profiler alternates reference/candidate order by pair and writes raw runs, per-backend summaries, paired rows, and a paired timing summary. It then performs a separate instrumented structural pass and writes `*_diagnostics.csv` plus `*_diagnostics_summary.csv`. Timing claims must use only the uninstrumented paired files.

If paired timing has already been completed, run only the inexpensive structural diagnostic:

```bash
python3 scripts/diagnose_u16_h2_two_hop_parent_shortcut.py \
  /path/to/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depths 16 32 64 \
  --foreground-connectivity 26 \
  --repeats 1 \
  --output profiles/v23_2_1_u16_h2_two_hop_parent_shortcut_diagnostics.csv
```

## Freeze criterion

Do not promote `parent-two-hop` merely because it avoids `find()` calls. Promote it only if:

1. ordered persistence output is exactly identical at d16/d32/d64;
2. successful-union, union-attempt, same-root, and zero-persistence counters remain invariant within every A/B pair;
3. paired local-sweep and wall-clock timings show a reproducible benefit at the production-relevant depths; and
4. the benefit is not explained by run order.
