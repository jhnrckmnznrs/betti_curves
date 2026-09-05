# v1.5 root-carrying union investigation

The v1.4 CX09T1 diagnostics showed that H2 spent a large fraction of its local-sweep work on union attempts whose endpoints were already in the same component. The conventional local persistence union also resolves both endpoints with `find()` for every representative edge.

v1.5 adds:

```text
--union-kernel <conventional|root-carrying>
```

The default remains `conventional` until the ablation is measured.

## Root-carrying rule

For one newly activated voxel, the conventional path repeatedly computes both

```text
find(current), find(neighbor)
```

for every representative neighbor. The root-carrying path keeps the surviving root returned by the previous merge and computes only `find(neighbor)` on the next edge. If rank-based linking changes the surviving root, the returned root is carried forward. The persistence birth/interface/outside action logic is shared with the conventional path.

Diagnostics add:

- `root_carry_attempts`: representative-edge union attempts handled by the root-carrying path;
- `avoided_find_calls`: calls to `find(current)` that were skipped because the current root was already known.

For a diagnostic root-carrying run with zero inactive-representative failures, these should satisfy

```text
root_carry_attempts == union_attempts
avoided_find_calls == root_carry_attempts
```

## Correctness and structural check

```bash
python3 validation/check_scalar_union_kernel_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

This requires exact persistence-multiset agreement among the in-memory scalar oracle, conventional stream, and root-carrying stream. It also requires the root-carrying path to report fewer `find()` calls.

## Balanced timing experiment

```bash
python3 scripts/profile_scalar_union_kernel.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_union_kernel_d16.csv
```

The profiler alternates conventional/root-carrying execution order and rejects any persistence mismatch.

## Operation-count comparison

```bash
python3 scripts/profile_scalar_union_diagnostics.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --output cx09t1_union_diagnostics_d16.csv
```

The useful decision variables are whole-wall and local-sweep speedup, the reduction in `find_calls`, and any change in `find_parent_steps`. Diagnostics are not used for timing because their counters add overhead.

## CX09T1 decision after v1.5

At slab depth 16 and 26-connectivity, root-carrying reduced `find()` calls from about 73.2M to 38.7M in H0 and from about 202.0M to 103.4M in H2 while preserving the exact interval multisets. H0 wall time was effectively neutral; H2 wall time improved consistently. v1.6 therefore promotes `root-carrying` to the scalar-stream default and moves the next H2 experiment to the direct-parent shortcut documented in `PARENT_SHORTCUT_INVESTIGATION.md`.
