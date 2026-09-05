# v1.12 H2 phase-boundary heap-trim investigation

## Motivation

The v1.10 packed local union-find was consistently faster for H2, but normal
runs showed about 50 MiB more process high-water RSS than `parent-rank`.
v1.11 established that the packed local state itself is smaller, not larger.
An external `/proc` sampler then showed that:

- the excess appears at the preparation-to-global-reduction transition;
- `MALLOC_ARENA_MAX=1` does not remove it;
- `MALLOC_TRIM_THRESHOLD_=0` removes the excess, but slows the whole run by
  about 12% because glibc trims repeatedly.

This is consistent with freed local-slab heap pages remaining resident until
the global reducer allocates its state.

## Ablation

v1.12 adds:

```text
--phase-trim off
--phase-trim before-reduce
```

`before-reduce` executes exactly one glibc `malloc_trim(0)` after
`prepare_h2_scalar_runs` returns and immediately before the global reduction
allocates its union-find state.

No persistence, pruning, union-find, interface, or run-file logic is changed.
The default remains `off` until the A/B benchmark is evaluated.

## Platform scope and unsafe-code policy

`malloc_trim` is a glibc extension. The option therefore requires a Linux GNU
(glibc) target and returns an explicit error elsewhere.

The crate previously used `#![forbid(unsafe_code)]`. v1.12 changes this to
`#![deny(unsafe_code)]` and grants `unsafe` permission only to the small
`allocator_trim` module. That module contains the audited FFI declaration and
one call to `malloc_trim(0)`. All topology, TIFF, persistence, pruning, and UF
modules remain covered by the crate-level `deny(unsafe_code)` policy.

## Profiling output

Each H2 scalar stream emits:

```text
PROFILE_TRIM scalar_h2_stream strategy=<...> attempted=<...> released=<...> seconds=<...>
```

The main A/B profiler intentionally does **not** enable `--h2-memory-audit`.
It uses `/usr/bin/time -v` externally because v1.11 showed that repeated
in-process memory auditing can itself perturb glibc residency behavior.

## Correctness

Run:

```bash
python3 validation/check_scalar_h2_phase_trim_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

The checker requires exact interval-multiset agreement among the in-memory H2
oracle and all four stream configurations:

```text
parent-rank + off
parent-rank + before-reduce
packed      + off
packed      + before-reduce
```

## Performance experiment

```bash
python3 scripts/profile_scalar_h2_phase_trim.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_h2_phase_trim_d16.csv
```

The target outcome is that `packed + before-reduce` keeps the packed UF speed
advantage while lowering peak RSS to roughly the parent-rank range, with only
a small one-time trim cost.
