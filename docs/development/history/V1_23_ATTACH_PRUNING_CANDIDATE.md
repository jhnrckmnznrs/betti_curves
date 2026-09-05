# v1.23 hierarchical H0 attach-pruning candidate

## Motivation

The v1.22 real-prefix profile on the 3792 x 3792 F32 synchrotron stack showed that hierarchical H0 has effectively bounded RAM but still propagates a large disk-backed attach stream through multiple fan-in levels. Doubling the prefix from 64 to 128 slices changed peak RSS only from about 1.93 to 2.04 GiB, while peak scratch increased from about 14.30 to 26.94 GiB and filesystem input operations increased about fourfold.

## Exact elder-dominated attach rule

Suppose an internal branch born at `b` attaches at filtration value `d` to a component that already reaches a surviving boundary terminal and whose current birth is `r`.

If `r <= b`, the finite persistence interval `[b,d)` is already determined and may be finalized immediately. Any topology outside the current block can only connect the terminal component to an equally old or older component, replacing `r` by some `r' <= r`; therefore the attaching branch remains the younger component at `d` in every extension of the block.

Only the case `b < r` can change the elder representative of the terminal component and therefore needs to propagate upward.

The same rule is applied both to replayed child attach events and to new terminal--internal unions formed while composing two summaries.

## Ablation

Reference:

```text
--h0-hier-attach-pruning off
```

Candidate:

```text
--h0-hier-attach-pruning elder-dominated
```

The candidate is intentionally limited to the disk-backed hierarchical H0 composition path. Leaf generation is unchanged; leaf attach events are pruned as soon as their first parent summary is constructed.

Each combine now reports:

```text
parent_attach_bytes=...
parent_interface_bytes=...
attach_finalized_early=...
attach_propagated=...
attach_pruning=...
```

and the final hierarchy profile reports aggregate early-finalized and propagated attach counts.

## Validation

Run:

```bash
scripts/check_v1_23_attach_pruning.sh examples/CX09T1 target/release/betti_curves
```

The gate stages an exact F32 CX09T1 subset, compares `off` and `elder-dominated` at multiple slab depths, requires exact persistence equality, and requires the candidate to reduce propagated attaches.

Then profile:

```bash
python3 scripts/profile_v1_23_attach_pruning.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slice-limit 0 \
  --slab-depths 8 16 32 \
  --repeats 3 \
  --output cx09t1_v1_23_attach_pruning.csv
```

If promoted, rerun the real-prefix profiler with `--attach-pruning elder-dominated` on 64 and 128 slices. The primary promotion metrics are exact persistence, aggregate parent attach bytes, peak scratch, filesystem I/O, and wall time.
