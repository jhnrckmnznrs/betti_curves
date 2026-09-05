# v1.22 real-F32 prefix profiling

The CX09T1 v1.22 profile validates exact terminal-free hierarchical H0 and the
`<=4A` pair frontier.  Before committing the 3792 x 3792 x 2048 synchrotron
volume, profile real prefixes to measure local-slab RSS and temporary-disk
scaling directly.

The profiler stages prefixes as symlinks; it does not copy TIFF pixels.  It
requires F32 source TIFFs, runs only the validated v1.22 hierarchical-stream
path, samples the dedicated `BETTI_TEMP_DIR` subtree, and aborts if scratch free
space falls below the configured safety floor.

Recommended first pass:

```bash
python3 scripts/profile_large_h0_hierarchical_prefix.py \
  /run/media/John/wTB16_2/SynchrotronImages/images/T1_S26/T1_S26_step0/Z0/slices \
  --binary target/release/betti_curves \
  --temp-dir /run/media/John/wTB16_2/betti_tmp \
  --slice-counts 64 128 \
  --slab-depths 8 \
  --foreground-connectivity 26 \
  --repeats 1 \
  --output t1_s26_h0_v122_prefix.csv
```

If d8 remains comfortably below the machine RAM and scratch limits, the next
comparison is `--slice-counts 128 --slab-depths 8 16`.
