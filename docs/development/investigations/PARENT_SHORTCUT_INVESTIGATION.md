# H2 direct-parent same-root investigation (v1.6)

The v1.5 root-carrying experiment reduced H2 `find()` calls by about half, but the remaining H2 sweep still spent substantial work resolving representative neighbors that were already in the current component. v1.6 tests a one-link sufficient condition before the remaining neighbor `find()`.

For a root-carrying activation, `current_root` is maintained as a proven union-find root. Before `find(neighbor)`, the ablation may test:

```text
neighbor == current_root || parent[neighbor] == current_root
```

Either condition proves that `neighbor` is already in the current component, so the union attempt is a same-root no-op and `find(neighbor)` can be skipped. Failure of the condition proves nothing; the implementation falls back to the unchanged exact root lookup.

The switch is:

```text
--neighbor-root-check find
--neighbor-root-check parent-shortcut
```

The shortcut is only active in the H2 `root-carrying` kernel. `find` is the default until representative profiling demonstrates a wall-time improvement.

Diagnostics add:

- `direct_parent_checks`: root-carrying H2 union attempts that tested the one-link condition;
- `direct_parent_hits`: attempts proved same-root without `find(neighbor)`;
- `avoided_neighbor_find_calls`: neighbor `find()` calls skipped by those hits.

For the shortcut path, `direct_parent_hits == avoided_neighbor_find_calls` must hold.

## Exact correctness check

```bash
python3 validation/check_scalar_h2_parent_shortcut_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

This compares the in-memory H2 scalar oracle, root-carrying `find`, and root-carrying `parent-shortcut` as exact persistence multisets. It also checks that the measured reduction in `find_calls` equals `avoided_neighbor_find_calls`.

## Balanced timing

```bash
python3 scripts/profile_scalar_h2_parent_shortcut.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_h2_parent_shortcut_d16.csv
```

## Operation counts

```bash
python3 scripts/profile_scalar_h2_parent_shortcut_diagnostics.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --output cx09t1_h2_parent_shortcut_diagnostics_d16.csv
```

The key decision variables are shortcut hit rate, reduction in `find_calls` and `find_parent_steps`, local-sweep speedup, and whole-wall speedup.


## CX09T1 result

At slab depth 16, the shortcut preserved all 220,093 H2 intervals exactly. It
resolved 61,886,137 of 98,554,512 root-carrying attempts directly (62.79%),
covering about 95.33% of same-root encounters. `find()` calls fell by about
59.9% and parent traversals by about 60.3%; the paired wall-time improvement
was about 2.7%. v1.7 therefore promotes `parent-shortcut` to the H2 default
while retaining `find` as the regression/reference path.
