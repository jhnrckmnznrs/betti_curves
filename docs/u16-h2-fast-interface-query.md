# U16 H2 fast root-invariant interface query (v23.6 production freeze)

## Motivation

The v23.4 d16/CX09T1 structural sweep audit recorded about 71.5 million
interface-representative queries in one native-U16 H2 leaf sweep. With
`--interface-state root-invariant`, the reference query derives `(z,
face_index)` from a local slab-linear root index using division and remainder
before deciding whether that root lies on the lower or upper slab face.

The sampled three-region timings from v23.4 are not used as optimization
evidence: each sampled region was only a tiny code fragment, and the near
one-third/one-third/one-third split indicates timing-call overhead dominated the
sample. The exact operation counters are the evidence used here.

## Candidate

`BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=1` replaces the division/remainder query
with an exactly equivalent range test (the upper-face start is precomputed once per slab):

- `index < face_size` means lower interface face;
- otherwise `index >= (depth - 1) * face_size` means upper interface face;
- every other valid local index is interior.

The upper-face local index is obtained by subtraction. No interface vector is
allocated and the root-invariant linking rule is unchanged.

As of the v23.6 production-freeze tree, the fast path is the production default.
Set `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=0` (also `false`, `off`, or `no`)
to restore the exact division/remainder reference path for regression or
ablation runs.

## Correctness gates

- Rust unit test compares the fast linear-index lookup with the existing
  division/remainder definition across multiple slab depths and face sizes.
- `validation/check_u16_h2_fast_interface_query_model.py` is an independent
  randomized arithmetic oracle.
- `validation/check_u16_h2_fast_interface_query_equivalence.py` compares exact
  ordered U16 H2 persistence rows at slab depths 16, 32, and 64 and also checks
  slab-depth interval-multiset invariance.

## Performance gate

`scripts/profile_u16_h2_fast_interface_query.py` uses interleaved paired A/B
runs. Both paths keep root dedup off, use `parent-shortcut`, and use
`root-invariant` interface state. Timing runs do not enable sweep diagnostics.
## v23.5.1 confirmation protocol

The first real-stack paired experiment showed a promising improvement at the
production-relevant slab depth 16, but the 11-pair d16 wall sign test was not
yet below 0.05. A focused confirmation therefore uses 31 d16 pairs, alternating
run order, with diagnostics disabled. The decision rule is fixed before the
confirmation data are observed.

All of the following are required to recommend promotion:

1. exact ordered persistence equivalence at slab depths 16, 32, and 64;
2. d16 median paired wall-time delta < 0%;
3. d16 median paired local-sweep delta < 0%;
4. at least 23 candidate wall-time wins among 31 non-tied pairs;
5. exact two-sided wall-time sign-test p < 0.05;
6. negative median wall-time delta in both order strata (reference-first and candidate-first).

Passing this gate produces a `PROMOTE_CANDIDATE` recommendation only. Changing
the default remains a separate freeze step so the raw measurements and exact
equivalence output can be reviewed first.


## v23.5.2 blocked confirmation protocol

The v23.5.1 focused run did not satisfy its pre-registered promotion gate. Its
paired median remained favorable, but the pair-to-pair spread was much larger
than the expected optimization effect. The candidate therefore stays off by
default while the confirmation design is changed rather than repeatedly
resampling the same unstable long sequence.

The v23.5.2 protocol is fixed before observing its data:

- slab depth 16 only;
- five blocks;
- seven measured A/B pairs per block (35 measured pairs total);
- one unmeasured warmup A/B pair at the beginning of each block;
- alternating reference/candidate order across successive pairs;
- 10 second cooldown between blocks;
- no sweep diagnostics during timing;
- optional Linux `perf stat` counters: cycles, instructions, branches, and
  branch misses;
- snapshots of `/proc/loadavg` and available CPU-frequency sysfs values before
  each measured run.

`perf` is automatically probed. If it is unavailable or disallowed by the host
kernel, the blocked timing experiment still runs and records
`perf_available=0`. Hardware counters are treated as corroborating evidence,
not as a hard promotion gate.

The timing recommendation requires all of the following:

1. exact ordered persistence equivalence at slab depths 16, 32, and 64;
2. exactly 35 measured pairs in the 5×7 design;
3. overall median paired wall-time delta < 0%;
4. overall median paired local-sweep delta < 0%;
5. at least 24 candidate wall-time wins among 35 non-tied pairs;
6. exact two-sided wall-time sign-test p < 0.05;
7. at least four of five blocks have a negative median wall-time delta;
8. both order strata have negative median wall-time deltas.

When perf is available, negative median cycle and instruction deltas are
reported as corroboration. The evaluator never changes the source default; a
separate reviewed freeze step is required after the raw measurements are
examined.


## v23.6 production decision

The pre-registered v23.5.2 blocked confirmation passed every promotion gate on
the representative CX09T1 U16 stack at slab depth 16:

- 5 blocks × 7 measured A/B pairs = 35 pairs;
- median paired wall-time delta: **-1.518526%**;
- median paired local-sweep delta: **-1.750983%**;
- candidate wall-time wins: **29/35**;
- exact two-sided wall sign-test: **p = 0.0001168419**;
- negative wall median in **4/5** blocks;
- reference-first median wall delta: **-1.126478%**;
- candidate-first median wall delta: **-1.747315%**.

Linux hardware counters were unavailable under the host's `perf_event_paranoid`
policy. They were pre-declared as corroborating evidence only and were not part
of the hard promotion gate. The raw results and decision record are archived
under `benchmarks/data/v23_5_2_confirmation/`.

The production implementation therefore uses the fast range/subtraction query
when the environment variable is unset. The old arithmetic path remains in the
same binary as an exact escape hatch.
