# v23.6 U16 H2 production freeze

Freeze date: 2026-09-06

This checkpoint freezes the benchmarked native-U16 `h2-scalar-hierarchical-stream` production profile after promotion of the fast root-invariant interface query. Cargo remains at `0.2.0`; `v23.6` is an internal production-freeze identifier, not a public semantic-version tag.

## Frozen production profile

- slab depth: `16`
- foreground connectivity: `26`
- local H2 birth state: `compact`
- global H2 birth state: `compact`
- global H2 UF layout: `packed`
- neighbor-root check: `parent-shortcut`
- interface state: `root-invariant`
- hierarchical cross storage: `direct`
- hierarchical outside structural pruning: `outside-dominated`
- native-U16 persistence keys: ON
- plateau-zero persistence elision: ON
- native-U16 H2 root dedup: OFF
- fast interface query: ON by default; set `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=0` for the exact legacy/reference path

## Promotion evidence

The pre-registered v23.5.2 blocked confirmation used 5 blocks × 7 measured d16 A/B pairs (35 pairs total). The candidate produced:

- median wall-time delta: `-1.52%`
- median local-sweep delta: `-1.75%`
- candidate wall-time wins: `29/35`
- exact two-sided wall sign-test: `p = 0.000116842`
- negative wall medians in `4/5` blocks
- negative wall medians in both A-first and B-first order strata

Raw confirmation data are preserved under `benchmarks/data/v23_5_2_confirmation/`.

## Frozen CX09T1 reference output

The production command in `V23_6_RUN_COMMANDS.txt` produced:

- source stack: `examples/CX09T1`
- dimensions: `274 × 274 × 448`
- voxel count: `33,634,048`
- H2 intervals: `220,093`
- schema: `birth,death`
- all intervals satisfy `birth < death`
- birth range: `135..259`
- death range: `136..265`
- maximum persistence: `86`
- output SHA-256: `6f08ac4bf5b2d155724d28b1d23dd6c4697ff53f707e5fd4786795f63a3744e6`

A copy is stored at `benchmarks/data/v23_6_freeze/v23_6_cx09t1_h2.csv`.

## Freeze gate

Run on the production machine after `./scripts/prepare_for_commit.sh`:

```bash
./scripts/check_u16_h2_fast_interface_query_v23_6_freeze.sh \
  ~/Desktop/stream_betti_curves_git/examples/CX09T1 \
  target/release/betti_curves
```

The expected final line is:

```text
PASS v23.6 U16 H2 fast-interface production freeze gate
```

The gate checks the independent arithmetic oracle, exact OFF-vs-ON persistence equivalence at d16/d32/d64, and default/unset vs explicit ON vs explicit legacy OFF behavior.
