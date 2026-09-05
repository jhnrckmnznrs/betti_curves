# v1.22 terminal-free hierarchical H0 root candidate

## Motivation

The v1.21 disk-backed hierarchy solved the slab-count-dependent RAM problem, but its final relative root summary remained unnecessarily expensive on disk. On the full CX09T1 F32 fixture the root attach run measured approximately 253 MiB at d8, 293 MiB at d16, and 313 MiB at d32. The root summary was then immediately reread by the whole-volume reducer.

The final block is the complete image domain, so it has no external spatial terminals. Therefore the final relative summary is not required.

## Candidate

For volumes with more than one leaf slab, v1.22 intercepts the final pair of adjacent hierarchical summaries before a root summary is written. It allocates the same bounded pairwise node set, constructs the exact shared-face cross run, and processes child attach/interface/cross events in the established deterministic order. The ordinary packed H0 persistence union-find is used because there are no surviving terminals. Finite intervals are appended directly to the finalized-pair run and the remaining root births become the essential intervals. Child and cross runs are then deleted.

The pairwise memory bound is unchanged: at most four face areas and 8 explicit bytes/node for native-F32 packed H0.

## Required diagnostics

A multi-slab optimized hierarchical run must report:

```text
root_materialized=false
root_attach_bytes=0
root_interface_bytes=0
final_interface_nodes=0
max_pair_nodes <= 4A
```

`PROFILE_H0_HIER_STREAM_FINAL` reports the actual terminal-free final-pair node count and packed UF allocation.

## Validation

Run:

```bash
scripts/check_v1_22_root_finalization.sh examples/CX09T1 target/release/betti_curves
```

Then profile:

```bash
python3 scripts/profile_v1_22_hierarchical_stream.py \
  examples/CX09T1 \
  --binary target/release/betti_curves \
  --slice-limit 0 \
  --slab-depths 8 16 32 \
  --repeats 3 \
  --output cx09t1_v1_22_hierarchical_stream.csv
```
