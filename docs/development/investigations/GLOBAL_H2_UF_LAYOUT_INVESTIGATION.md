# Global H2 union-find layout investigation (v1.15)

## Motivation

v1.14 promoted compact global H2 birth storage, reducing the measured explicit global state from about 37 bytes/node to 13 bytes/node. The remaining global union-find state is:

- parent: 4 bytes/node,
- rank: 1 byte/node,
- compact birth: 8 bytes/node.

v1.15 tests removing the separate rank vector by encoding root rank directly in the parent word.

## Packed global root encoding

The packed global layout reserves only the top 256 `u32` words for root markers. A root of rank `r` stores `u32::MAX - r`; every non-root stores its parent node ID. This preserves almost the full 32-bit node-ID range, unlike the local packed representation that reserves the high bit. The packed global state therefore supports up to `u32::MAX - 255` nodes.

The candidate changes explicit compact global H2 state from approximately 13 bytes/node to 12 bytes/node.

For the stress-test volume `3792 x 3792 x 2048`:

| slab depth | interface nodes | compact parent-rank | compact packed |
|---:|---:|---:|---:|
| 32 | 1,840,545,792 | ~22.28 GiB | ~20.57 GiB |
| 40 | 1,495,443,456 | ~18.11 GiB | ~16.71 GiB |
| 48 | 1,236,616,704 | ~14.97 GiB | ~13.82 GiB |
| 64 | 920,272,896 | ~11.14 GiB | ~10.28 GiB |

The savings are smaller than v1.14 but still exceed 1 GiB at practical slab depths for the large volume.

## Ablation

The v1.15 default remains:

```text
--global-h2-birth-state compact
--global-h2-uf-layout parent-rank
```

The candidate is:

```text
--global-h2-birth-state compact
--global-h2-uf-layout packed
```

Validate exact persistence with:

```bash
python3 validation/check_scalar_h2_global_uf_layout_equivalence.py \
  examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

Then profile:

```bash
python3 scripts/profile_scalar_h2_global_uf_layout.py \
  examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_h2_global_uf_layout_d16.csv
```

Promotion requires exact persistence equality, zero separate rank allocation in the packed path, and no material wall-time regression.
