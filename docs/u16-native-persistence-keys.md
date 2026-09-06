# Native U16 persistence keys (v20 experimental)

The v19 bounded hierarchical persistence reconciler removed the largest global
H2 bottleneck, but integer TIFF stacks still widened every U16 sample to the
canonical 8-byte `ScalarKey`. v20 adds a compact exact key for the hot U16
persistence path while preserving the existing wide representation as a
same-binary reference.

## Representation

`U16Key` is a four-byte `u32` wrapper:

- `0..=65535` represent source U16 intensities exactly;
- `65536` is reserved for the compact H2 Outside marker.

The ordering of all finite keys is exactly the U16 numerical ordering. When a
value leaves the local/native persistence path, `U16Key::widen()` reconstructs
the canonical `ScalarKey`, so CSV output and cross-backend interval semantics
remain unchanged.

The compact path is enabled by default for U16 hierarchical persistence. Set

```bash
BETTI_PERSIST_U16_NATIVE_KEYS=0
```

to restore the v19 wide64 U16 path in the same binary.

## What changes

The following U16 data become four-byte keys:

- decoded slab/birth storage;
- leaf boundary-face values;
- attach/interface/cross event values;
- hierarchical summary runs;
- finalized finite-pair runs.

The topology algorithms, deterministic event priorities, interface
sparsification, and hierarchical reducers are shared with v19.

## Plateau audit

The direct leaf kernels now count finite merge pairs discarded because
`birth == death`. Native U16 hierarchical runs report

```text
PROFILE_U16_PERSIST_LEAF ... zero_persistence_pairs_elided=...
```

No plateau semantics are changed in v20. The counter is intended to decide
whether a later optimization should prevent these zero-length pairs from being
constructed in the first place.

## Validation

Use

```bash
python3 validation/check_u16_native_key_persistence_equivalence.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26 \
  --dimensions h0 h2
```

The validator compares exact interval multisets from native32 and forced-wide64
hierarchical runs.
