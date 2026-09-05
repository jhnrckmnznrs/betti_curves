# End-to-end native F32 H0 storage investigation (v1.18 candidate)

## Motivation

The v1.17 scalar H0 stream already used native four-byte `F32Key` values inside
F32 slabs, but widened slab summaries back to the canonical eight-byte
`ScalarKey` before cross-interface processing, temporary-run serialization, and
global reconciliation. On the target `3792 x 3792 x 2048` F32 volume, that
boundary dominates both disk storage and global H0 memory.

v1.18 tests retaining exact native F32 keys through the complete H0 streaming
pipeline and widening only when final CSV values are emitted.

## Representation

For F32 input with `--f32-key-mode native32`, H0 now keeps `F32Key` through:

1. local slab persistence summaries,
2. boundary-face values,
3. cross-slab interface sparsification,
4. disk-backed local-pair, attach, interface, cross, and birth records,
5. global H0 union-find birth state.

The final persistence CSV is unchanged: exact F32 keys are widened losslessly
to the canonical `ScalarKey` only when a birth/death value is printed.

`--f32-key-mode legacy64` remains the same-binary reference path. U8/U16/F64
input continues to use the canonical eight-byte scalar representation.

## Record widths

| state / record | legacy64 | native32 |
|---|---:|---:|
| interface birth | 8 B | 4 B |
| finalized finite pair | 16 B | 8 B |
| interface/cross pair event | 16 B | 12 B |
| attach event | 20 B | 12 B |
| global H0 parent | 4 B/node | 4 B/node |
| global H0 rank | 1 B/node | 1 B/node |
| global H0 birth | 8 B/node | 4 B/node |
| explicit global H0 total | 13 B/node | 9 B/node |

Thus the deterministic explicit global H0 state is reduced by `4/13 = 30.77%`
for F32 input before allocator overhead.

## Target-volume scaling

For `3792 x 3792 x 2048`, the current interface-node plan gives:

| slab depth | interface nodes | legacy64 global H0 | native32 global H0 |
|---:|---:|---:|---:|
| 16 | 3,681,091,584 | 44.57 GiB | 30.85 GiB |
| 24 | 2,473,233,408 | 29.94 GiB | 20.73 GiB |
| 32 | 1,840,545,792 | 22.28 GiB | 15.43 GiB |
| 40 | 1,495,443,456 | 18.11 GiB | 12.53 GiB |
| 48 | 1,236,616,704 | 14.97 GiB | 10.37 GiB |
| 64 | 920,272,896 | 11.14 GiB | 7.71 GiB |

At depth 32, the interface-birth file alone falls from 13.71 GiB to 6.86 GiB.
The other run files also shrink because native pair and attach records are
narrower; their exact total depends on topology and is reported after slab
preparation as `PROFILE_H0_STORAGE ... total_run_bytes=...`.

## Guarded validation

Build first, then validate a two-slice subset of a real F32 stack with two
one-slice slabs. This exercises local persistence, a cross-slab interface, disk
runs, and global reconciliation while keeping the independent in-memory oracle
tractable:

```bash
python3 validation/check_scalar_h0_f32_end_to_end_equivalence.py \
  /path/to/f32_stack \
  --binary target/release/betti_curves \
  --slice-limit 2 \
  --slab-depth 1 \
  --foreground-connectivity 26
```

The guard requires:

- exact interval-multiset equality with the independent in-memory H0 path,
- exact equality between `legacy64` and `native32`,
- `source pixel type: F32`,
- eight-byte disk keys for `legacy64`,
- four-byte disk keys for `native32`,
- interface-birth file bytes equal to `interface_nodes * disk_key_bytes`,
- global-state accounting equal to `interface_nodes * (4 + 1 + key_bytes)`.

Balanced profiling remains available through `scripts/profile_scalar_f32_keys.py`.
For the large stack, use `--slice-limit 2 --slab-depth 1 --modes h0-scalar-stream`
for a guarded A/B before committing to the full-volume run.

## Promotion rule

Do not promote this candidate from code inspection alone. Promotion requires a
successful Rust test/build/Clippy gate, exact real-F32 equivalence, and observed
four-byte H0 disk storage. Full-volume profiling should then begin at slab depth
32 on the 32 GiB machine, with `BETTI_TEMP_DIR` pointed at a filesystem with
ample free space.

H2 is unchanged by this candidate.
