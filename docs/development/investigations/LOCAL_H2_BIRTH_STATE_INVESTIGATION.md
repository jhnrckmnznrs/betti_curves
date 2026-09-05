# Local H2 birth-state compaction (v1.16)

## Motivation

After v1.14 reduced the global H2 birth state and v1.15 investigated global rank packing,
the next limiting allocation for very large F32 slabs is the local background-component birth
array. The reference representation stores one Rust enum per voxel:

```text
Outside | Finite(F32Key)
```

On the current target ABI this occupies 8 bytes per F32 voxel, while the finite key itself is
only 4 bytes.

## Candidate representation

`--local-h2-birth-state compact` stores one `F32Key`/`ScalarKey` per voxel and reserves the
ordered encoding of positive infinity as the outside marker. Scalar TIFF decoding already
rejects all non-finite samples, so this marker cannot collide with accepted image data. It is
also greater than every finite ordered key, matching the elder-rule convention that the
outside component is older than every finite background component.

The reference path remains available as:

```text
--local-h2-birth-state tagged
```

v1.17 promotes `compact` to the production default after exact-equivalence and balanced real-stack profiling. The `tagged` representation remains available as the reference/debug backend.

## Expected F32 memory effect

For a slab with N voxels and the current packed local UF / parent-sentinel active state:

```text
reference local UF state: parent 4N + birth 8N = 12N bytes
compact local UF state:   parent 4N + birth 4N =  8N bytes
```

Thus compact births reduce the explicit local UF state by about one third and remove 4 bytes
per slab voxel.

For a 3792 x 3792 cross-section:

- depth 48: 690,204,672 voxels, about 2.57 GiB saved;
- depth 64: 920,272,896 voxels, about 3.43 GiB saved.

These figures exclude the input-value vector, scalar-order vector, event buffers, boundary-face
state, allocator overhead, and the later global reduction state.

## Validation

Run:

```bash
python3 validation/check_scalar_h2_local_birth_state_equivalence.py \
  examples/CX09T1/ --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26
```

The checker compares both local birth strategies against the independent in-memory H2 scalar
implementation and requires exact persistence-multiset equality. The v1.16.2 harness also forces
`--f32-key-mode native32`, records the reported source pixel type/effective configuration, and
fails if an F32 input reports any local key width other than 4 bytes. This prevents a wide-key
run from being mistaken for the intended native-F32 compaction experiment.

Profile with:

```bash
python3 scripts/profile_scalar_h2_local_birth_state.py \
  examples/CX09T1/ --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26 --repeats 5 \
  --output cx09t1_h2_local_birth_state_d16.csv
```


## v1.17 promotion result

The guarded benchmark on CX09T1 (slab depth 16, five paired repeats) reported the source pixel type as `U16`, so the F32-only native32 optimization was correctly not involved in this particular promotion measurement. Both strategies used `key_bytes=8`.

| metric (median) | tagged | compact | change |
|---|---:|---:|---:|
| wall time | 11.747 s | 9.044 s | lower |
| local sweep | 8.064 s | 6.165 s | lower |
| peak RSS | 93.26 MiB | 84.33 MiB | -9.8% |
| local birth storage | 18.33 MiB | 9.16 MiB | -50.0% |
| total explicit local H2 state | 22.91 MiB | 13.75 MiB | -40.0% |

All ten runs produced 220,093 intervals with one canonical persistence hash, and compact won all five paired wall-time comparisons. This is sufficient for production promotion; the tagged implementation is retained to serve as an equivalence oracle during future changes.
