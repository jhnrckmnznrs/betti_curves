# Independent validation

The validation suite asks separate questions. Agreement between two production
paths is useful, but it is not enough when both paths may share a mistake.

## 1. Native Rust tests

```bash
cargo test --all-targets
```

These tests cover local union rules, dual connectivity, exact scalar ties,
signed zero, cross-face prefix partitions, atomic output preservation, size
planning, outside handling, and elder-rule plateau behavior.

The cross-face test exhausts every binary mask pair on a 2-by-2 face and checks
the connected-component partition after each filtration prefix. It therefore
tests the property needed by persistence, not only the final forest size.

## 2. Python unit tests

```bash
python3 -m unittest discover -s python_scripts/tests -v
python3 -m unittest discover -s validation -p 'test_*.py' -v
```

The first command tests step-curve parsing, domains, right-continuity,
normalization, triangular matrix storage, duplicate labels, and atomic output.

The second tests separately written H0 and H2 graph-barcode sweeps. It checks
that interval coverage equals independently labeled component counts and that
branch node files form a valid rooted object with the same interval multiset.

## 3. Direct cubical boundary matrices

```bash
python3 validation/tiny_cubical_oracle.py
```

This oracle does not use the background-component formula to compute homology.
It constructs cubical chain groups, builds boundary matrices over the field
with two elements, and calculates ranks. It checks all 256 binary 2-by-2-by-2
images for each of the two cubical constructions, plus a 3-by-3-by-3 shell with
nonzero H2.

Expected summary:

```text
PASS: 512 tiny construction checks and two direct shell H2 checks
```

## 4. Compiled black-box campaign

Install dependencies and build first:

```bash
python3 -m pip install -r validation/requirements.txt
cargo build
python3 validation/reference_check.py
```

Set `BETTI_CURVES_BINARY` to validate a different executable:

```bash
BETTI_CURVES_BINARY=target/release/betti_curves \
  python3 validation/reference_check.py
```

`reference_check.py` creates temporary TIFF stacks and invokes the executable
through its public command line. It covers:

- SciPy-labeled Betti-0 foreground components of `I <= t`;
- SciPy-labeled Betti-2 enclosed components of `I > t`;
- 6/26 and 26/6 dual connectivity;
- slab depths below, equal to, and above the image depth;
- threshold-wise, event, disk-backed, persistence, exact scalar, batch, and
  branch-tree paths;
- exact interval multisets from independent elder-rule graph sweeps;
- negative and fractional floating values and signed zero;
- natural numeric slice order and recursive batch hierarchy;
- shell, diagonal-contact, boundary, and random small-volume cases.

The random seed is fixed. Temporary stacks and outputs are removed on exit.

## 5. Optimization regression and profiling

After scalar-stream implementation changes, run the complete suite through:

```bash
scripts/check_optimized_scalar_stream.sh
```

For one representative stack that fits both scalar implementations, require
exact stream/in-memory interval-multiset equality:

```bash
python3 validation/check_scalar_stream_equivalence.py /data/stack \
  --binary target/release/betti_curves --slab-depth 8 \
  --foreground-connectivity 26
```

For performance A/B testing, preserve a baseline executable before rebuilding
and then run:

```bash
python3 scripts/profile_scalar_stream.py /data/stack \
  --baseline /path/to/betti_curves_baseline \
  --candidate target/release/betti_curves --repeats 3
```

The profiler canonicalizes interval CSVs as multisets and checks their SHA-256
hashes before presenting speedups. It also records `/usr/bin/time -v` peak RSS
and the candidate's internal prepare/reduce/cleanup phase timings.

## 6. What remains outside this suite

These tests establish strong small-volume exactness. They do not by themselves
establish peak resident memory on a target machine, TIFF metadata correctness
outside the stated input contract, crash recovery of temporary runs, or the
future associativity theorem needed for bounded global slab summaries. Those
claims require separate tests and, where appropriate, proofs.


## Scalar stream optimization ablations

For one representative scalar stack, check a selected streaming configuration
against the in-memory exact persistence implementation:

```bash
python3 validation/check_scalar_stream_equivalence.py /data/stack \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26 \
  --merge-strategy scan --interface-order radix --event-order verify
```

For the full 2 x 2 x 2 optimization matrix:

```bash
python3 validation/check_scalar_stream_ablation_equivalence.py /data/stack \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26
```

Every stream cell is compared as an exact interval multiset with the independent
in-memory scalar implementation.


For an F32 stack, separately verify the native-width representation against
both the legacy-wide stream and the in-memory exact oracle:

```bash
python3 validation/check_scalar_f32_key_equivalence.py /data/f32_stack \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26
```

## v1.3 neighbor-kernel equivalence

After building the release binary, compare the new interior-neighbor path against both the generic streaming path and the independent in-memory scalar persistence implementation:

```bash
python3 validation/check_scalar_neighbor_kernel_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

For phase-level profiling and balanced generic/interior-fast timing, use `scripts/profile_scalar_neighbor_kernel.py`; see `PREPARATION_PROFILING.md`.


## v1.6 H2 parent-shortcut equivalence

Verify the direct-parent same-root shortcut against both the root-carrying `find` reference and the independent in-memory H2 scalar persistence path:

```bash
python3 validation/check_scalar_h2_parent_shortcut_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 \
  --foreground-connectivity 26
```

Balanced timing and operation-count tools are `scripts/profile_scalar_h2_parent_shortcut.py` and `scripts/profile_scalar_h2_parent_shortcut_diagnostics.py`.


## v1.8 active-state equivalence

Verify the parent-sentinel activity representation against both the historical separate byte array and the independent in-memory scalar persistence path:

```bash
python3 validation/check_scalar_active_state_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves \
  --slab-depth 16 --foreground-connectivity 26
```

Balanced timing/RSS and operation-count tools are `scripts/profile_scalar_active_state.py` and `scripts/profile_scalar_active_state_diagnostics.py`.


### H2 phase-trim equivalence (v1.12)

```bash
python3 validation/check_scalar_h2_phase_trim_equivalence.py <stack> \
  --binary target/release/betti_curves --slab-depth 16 --foreground-connectivity 26
```

This compares the in-memory H2 scalar persistence multiset with parent-rank
and packed streaming UF layouts, each with phase trimming off and with one
`before-reduce` trim.

## Production-default validation

`check_production_scalar_equivalence.py` is the release-level real-stack
checker. It invokes `h0-scalar-stream` and `h2-scalar-stream` **without tuning
flags**, verifies their reported production configuration, compares them to the
in-memory scalar oracle, and repeats the stream check at multiple slab depths.
This catches both topology regressions and accidental default drift.

Example:

```bash
python3 validation/check_production_scalar_equivalence.py /path/to/stack \
  --binary target/release/betti_curves --slab-depths 8 16 32
```

## v1.16 local-H2 birth-state equivalence

Verify compact local background-component births against both the tagged local
reference and the independent in-memory H2 scalar persistence implementation:

```bash
python3 validation/check_scalar_h2_local_birth_state_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves --slab-depth 16 \
  --foreground-connectivity 26
```

Use `scripts/profile_scalar_h2_local_birth_state.py` for the balanced external
wall/RSS ablation. The stream emits `PROFILE_LOCAL_H2_STATE` without enabling
the intrusive memory audit.

## v1.20 architecture-candidate equivalence

The aggregate v1.20 gate keeps every new architecture path experimental until
it matches the independent scalar implementations:

```bash
scripts/check_v1_20_architecture_candidate.sh examples/CX09T1 target/release/betti_curves
```

It checks hierarchical H0 at several slab depths on native CX09T1, stages an
exact-value F32 subset to check H0 birth-buffer reuse/direct event sinks and H2
native-F32 end-to-end storage, and checks `--merge-strategy auto` against both
scan and heap. The F32 staging helper accepts only U8/U16/F32 sources so the
conversion is exact.

## v1.21 optimized hierarchical H0

`check_scalar_h0_hierarchical_stream_equivalence.py` stages an exact-value F32 CX09T1 fixture and compares in-memory H0, flat native32/direct streaming H0, and `h0-scalar-hierarchical-stream`. In addition to exact persistence equality, it checks the measured pair frontier is at most four face areas, the final interface state is two face areas for a multi-slice fixture, and the disk key width is four bytes.


## v23.6 fast-interface production freeze

The promoted native-U16 H2 fast root-invariant interface query is default-on.
The release gate verifies both the old and new arithmetic paths and separately
checks default behavior so an accidental environment-toggle regression cannot
pass unnoticed.

After `scripts/prepare_for_commit.sh`, run:

```bash
scripts/check_u16_h2_fast_interface_query_v23_6_freeze.sh \
  examples/CX09T1 \
  target/release/betti_curves
```

This runs the independent arithmetic oracle, exact ordered OFF-vs-ON persistence
equivalence at slab depths 16/32/64, and a d16 default-state check requiring:

- environment variable unset => `h2_fast_interface_query=on`;
- explicit `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=1` => on;
- explicit `BETTI_PERSIST_H2_FAST_INTERFACE_QUERY=0` => off;
- identical ordered persistence rows for default, explicit-on, and reference-off.
