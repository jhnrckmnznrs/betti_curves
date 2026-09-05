# v1.24 candidate — outside-aware hierarchical H2 stream

## Goal

v1.24 applies the successful large-volume H0 architecture to scalar H2 while
preserving H2's distinguished outside component.  The candidate is experimental;
the existing flat `h2-scalar-stream` path remains the reference implementation.

The new mode is:

```text
h2-scalar-hierarchical-stream
```

For the first candidate it requires F32 input and the explicit representation
settings:

```text
--f32-key-mode native32
--local-h2-birth-state compact
--global-h2-birth-state compact
--global-h2-uf-layout packed
```

## Architecture

Each leaf slab is reduced relative to its two z-faces and the distinguished
outside class.  The leaf path uses all currently validated large-volume H2
storage optimizations together:

- native 4-byte `F32Key` storage end-to-end;
- reuse of the decoded F32 slab buffer as compact local H2 birth storage;
- direct local persistence-event sinks instead of whole-slab event vectors;
- implicit boundary node IDs in the direct hierarchical leaf (only face scalar values are retained);
- compact local outside-marker representation;
- elder-dominated H2 attach finalization.

Adjacent summaries are combined online with binary fan-in.  Their touching
z-faces become interior; only the two outer z-faces remain as finite terminals.
Pair birth arrays are built directly by concatenating child face values rather
than initializing and overwriting a full pair-state buffer.
Outside events remain explicit summary events.  Intermediate outside-connected
roots use the reserved outside birth marker, so the outside class is always
older than every finite background component.

At equal superlevel value each fan-in processes events in the same deterministic
priority as the flat global H2 reducer:

```text
outside -> attach -> interface -> cross-interface
```

The final fan-in is terminal-free.  It uses the compact packed global H2 union-
find with one structural outside node, emits the remaining finite H2 intervals
directly, and rejects any computation that ends with a finite background root
not connected to outside.  No root relative summary is materialized.

## Exact elder-dominated H2 attach pruning

For a superlevel component attached to a surviving terminal, let `r` be the
current finite terminal-component birth and `b` the attaching branch birth.
The elder component has the larger birth.  Therefore:

- if the terminal component is already outside, the finite branch is final;
- if `r >= b`, the finite branch is already the younger class and `[d,b)` is
  final at the current merge value `d`;
- only `b > r` can change the terminal component's future elder birth, so only
  that attach must be propagated upward.

Future continuation outside the current block can only make a boundary-connected
component older, never younger, so an attach finalized by this rule cannot be
reversed by a later fan-in.

The rule is applied both in optimized leaves and in intermediate fan-ins.

## Memory bound

A pairwise fan-in contains at most the two faces of each child, hence at most
four face areas.  With native F32 compact births and packed parent/rank, the
explicit pairwise H2 state is 8 bytes per node:

```text
packed parent/rank  4 bytes
F32 birth marker    4 bytes
---------------------------
                    8 bytes/node
```

For a `3792 x 3792` face this gives

```text
4A = 57,517,056 nodes
explicit pair state ~= 438.8 MiB
```

independent of the number of z-slabs.  The final structural outside node adds
one packed parent word.

## Outside invariant

Intermediate fan-ins may contain more than one root whose birth is the outside
marker.  These roots represent portions of the same semantic distinguished
outside class that are not required to be structurally joined inside the
current relative block.  Two outside-marked roots never produce a finite pair.
If such a root no longer touches a surviving z-terminal, it carries no future
finite persistence information and can be discarded.  Propagated outside
events reconstruct outside connectivity for surviving terminals.

The terminal-free final fan-in restores one structural distinguished outside
node and requires all finite roots to join it.

## Independent algebra check

Before packaging, the summary algebra was reimplemented independently and
compared with an unsplit distinguished-outside superlevel computation on 13,600
random small 3-D volumes.  The tests covered:

- 6- and 26-connectivity;
- tied scalar values;
- several shapes and slab depths;
- odd numbers of slabs and multilevel fan-in;
- elder-dominated H2 attach pruning.

All 13,600 persistence multisets agreed.  This is not a substitute for compiling
the Rust implementation or the CX09T1 gate.

## Workstation acceptance gate

First compile the candidate:

```bash
cargo fmt --all
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release --locked
```

Then run:

```bash
scripts/check_v1_24_h2_hierarchical_stream.sh \
  examples/CX09T1 \
  target/release/betti_curves
```

The gate stages an exact F32 subset of CX09T1 and requires exact persistence-
multiset equality among:

1. independent in-memory scalar H2;
2. flat native32/compact/packed H2 streaming;
3. hierarchical H2 at slab depths 4, 8, and 16.

It additionally requires:

- `disk_key_bytes=4`;
- compact global H2 birth storage;
- packed global H2 UF;
- `max_pair_nodes <= 4A`;
- packed/native32 pair-state accounting;
- `root_materialized=false`;
- zero root attach/outside/interface bytes;
- zero final interface nodes.

## Performance profile

After the exactness gate passes:

```bash
python3 scripts/profile_v1_24_h2_hierarchical_stream.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slice-limit 0 \
  --slab-depths 8 16 32 \
  --foreground-connectivity 26 \
  --repeats 3 \
  --output cx09t1_v1_24_h2_hierarchical_stream.csv
```

Promotion requires exact canonical persistence in every paired run, a fixed
pairwise frontier, and a meaningful memory/I/O advantage without an unacceptable
runtime regression.  The flat H2 stream remains available regardless of the
outcome.
