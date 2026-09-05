# Global H0 union-find layout investigation (v1.19)

## Motivation

The v1.18 end-to-end native-F32 H0 path removes the 8-byte scalar-key widening
from disk runs and global birth state. On the real two-slice T1_S26 profile it
preserved the exact persistence multiset while reducing median wall time from
23.43 s to 19.95 s, peak RSS from about 1013 MiB to 688 MiB, temporary-run
bytes by 30.0%, and explicit global H0 state by 30.77%.

After that change the native-F32 global H0 union-find still stores:

- parent: 4 bytes/interface node,
- rank: 1 byte/interface node,
- birth: 4 bytes/interface node.

v1.19 tests removing the separate rank byte.

## Packed root encoding

`--global-h0-uf-layout packed` uses the same conservative root-marker design as
the existing global H2 packed ablation. Non-roots store an ordinary `u32`
parent node ID. Roots store `u32::MAX - rank`, reserving only the top 256 words
for rank markers.

The packed layout therefore supports at most `u32::MAX - 255` interface nodes.
The target `3792 x 3792 x 2048` volume has about 1.84 billion H0 interface
nodes at slab depth 32, well inside that limit.

For native F32 input the explicit global state changes from 9 to 8 bytes/node:

| slab depth | interface nodes | native32 parent-rank | native32 packed | saved |
|---:|---:|---:|---:|---:|
| 16 | 3,681,091,584 | 30.85 GiB | 27.43 GiB | 3.43 GiB |
| 32 | 1,840,545,792 | 15.43 GiB | 13.71 GiB | 1.71 GiB |
| 40 | 1,495,443,456 | 12.53 GiB | 11.14 GiB | 1.39 GiB |
| 48 | 1,236,616,704 | 10.37 GiB | 9.21 GiB | 1.15 GiB |
| 64 | 920,272,896 | 7.71 GiB | 6.86 GiB | 0.86 GiB |

This is an 11.11% reduction in explicit global H0 state on top of the v1.18
native-F32 reduction. Disk-run sizes are unchanged by this ablation.

## Candidate and reference

Reference:

```text
--f32-key-mode native32
--global-h0-uf-layout parent-rank
```

Candidate:

```text
--f32-key-mode native32
--global-h0-uf-layout packed
```

The real-F32 validation/profile campaign completed successfully, so v1.19 promotes
`packed` to the production default. `parent-rank` remains available as the explicit
reference/debug backend.

## Validation

```bash
BETTI_TEMP_DIR=/path/to/large-temp \
python3 validation/check_scalar_h0_global_uf_layout_equivalence.py \
  /path/to/T1_S26_step0/Z0/slices \
  --binary target/release/betti_curves \
  --slice-limit 2 \
  --slab-depth 1 \
  --foreground-connectivity 26
```

The validator requires exact persistence equality among the in-memory H0
oracle, native32 parent-rank stream, and native32 packed stream. It also checks
that the packed run reports zero global rank allocation.

## Balanced profile

```bash
BETTI_TEMP_DIR=/path/to/large-temp \
python3 scripts/profile_scalar_h0_global_uf_layout.py \
  /path/to/T1_S26_step0/Z0/slices \
  --binary target/release/betti_curves \
  --slice-limit 2 \
  --slab-depth 1 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output t1_s26_h0_global_uf_layout.csv
```

Promotion result on the two-slice T1_S26 benchmark: all 10 runs produced exactly
374,303 intervals with the same canonical hash; packed allocated zero rank bytes and
reduced explicit global state from 246.84 MiB to 219.41 MiB. Median reduction time
changed from 4.525 s to 4.253 s (about -6.0%) and median wall time from 20.954 s to
20.815 s. Peak RSS was effectively unchanged because local preparation dominated the
small two-slice process peak. The deterministic global-state reduction is 1 byte per
interface node (11.11% of native32 global H0 state), so packed is promoted.
