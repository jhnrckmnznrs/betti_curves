# Global H2 birth-state investigation (v1.14)

## Motivation

The local slab kernel is no longer the only limiting memory phase for very large scalar volumes.  For a streamed H2 computation, every slab boundary contributes global interface nodes that remain represented during the final disk-backed reconciliation.  The historical scalar H2 reducer stores three arrays per global node:

- `parent: u32` (4 bytes),
- `rank: u8` (1 byte),
- `BackgroundBirth`, a tagged `Outside | Finite(ScalarKey)` enum (16 bytes per element on the supported 64-bit target).

The finite interface births are first read from disk as `Vec<ScalarKey>` and converted into a tagged vector. The constructor then appends the outside node. On CX09T1 this final push caused the tagged birth vector capacity to double, so the measured allocated birth capacity was about 32 bytes per interface node rather than 16. The measured explicit global state was therefore about 37 bytes/node. The compact path avoids both the enum and this capacity-doubling construction.

For the stress-test volume `3792 x 3792 x 2048`, slab depth 32 produces 1,840,545,792 global interface nodes. Extrapolating the measured 37-byte/node tagged allocation gives roughly 63.4 GiB of explicit global-UF capacity before reader buffers and allocator overhead. This cannot fit on a 32 GiB machine.

## Compact representation

The new `--global-h2-birth-state compact` path stores finite births directly as `Vec<ScalarKey>`.  It does not encode an outside tag for every node.  Instead, the unique outside component is represented structurally: the distinguished `outside_node` is forced to remain the union-find root whenever a finite component joins the outside component.

This preserves the H2 elder rule:

- outside is older than every finite background component;
- finite/finite merges retain `max(birth_a, birth_b)` and pair against `min(birth_a, birth_b)`;
- finite/outside merges pair the finite birth and keep the outside component;
- branch attachment to outside pairs the branch birth without altering finite state.

The input `Vec<ScalarKey>` is moved directly into the compact UF, so no second birth allocation is constructed.

Approximate explicit state per global interface node becomes:

- parent: 4 bytes,
- rank: 1 byte,
- finite birth: 8 bytes,
- total: about 13 bytes/node.

The rank vector remains unchanged in v1.14.  Packing or otherwise reducing global rank/parent state is deliberately deferred until the birth-state change is validated independently.

## Large-volume estimates

For `3792 x 3792 x 2048`, ignoring small outside-node and allocator overhead:

| slab depth | slabs | interface nodes | measured-tagged extrapolation | compact |
|---:|---:|---:|---:|---:|
| 32 | 64 | 1,840,545,792 | ~63.42 GiB | ~22.28 GiB |
| 40 | 52 | 1,495,443,456 | ~51.53 GiB | ~18.11 GiB |
| 48 | 43 | 1,236,616,704 | ~42.61 GiB | ~14.97 GiB |
| 64 | 32 | 920,272,896 | ~31.71 GiB | ~11.14 GiB |
| 96 | 22 | 632,687,616 | ~21.80 GiB | ~7.66 GiB |

These are global-reduction state estimates only.  A usable slab depth must also fit the local preparation phase, so the optimum on a 32 GiB machine remains an empirical balance.

## Ablation

CX09T1 v1.14 profiling produced exact persistence equality in all ten measured runs. Compact reduced explicit global state from 155,557,477 bytes to 54,655,333 bytes (64.9%), reduced median peak RSS by about 6.5%, reduced H2 reduction time by about 7.1%, and reduced median wall time by about 1.8%. Compact is therefore promoted in v1.15. The historical tagged implementation remains available as the reference path.

```text
--global-h2-birth-state compact   # production default in v1.15
--global-h2-birth-state tagged    # historical reference
```

Every H2 reduction emits:

```text
PROFILE_GLOBAL_H2_STATE strategy=... nodes=... parent_bytes=... rank_bytes=... birth_bytes=... total_bytes=...
```

Use:

```bash
python3 validation/check_scalar_h2_global_birth_state_equivalence.py \
  examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

and then:

```bash
python3 scripts/profile_scalar_h2_global_birth_state.py \
  examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --repeats 5 \
  --output cx09t1_h2_global_birth_state_d16.csv
```

Promotion requires exact persistence equality and a meaningful RSS reduction without a material wall-time regression.
