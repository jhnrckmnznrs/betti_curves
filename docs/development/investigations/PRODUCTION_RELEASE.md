# Production scalar-stream release (v1.19)

v1.19 freezes the measured scalar-stream configuration after the focused H0/H2
optimization campaign and the real-F32 large-volume H0 storage campaign. Historical alternatives remain available as explicit
ablation/reference switches, but production profiling must invoke the stream
commands with **no tuning flags**.

## Frozen default policy

### H0

- `merge_strategy=scan`
- `interface_order=radix`
- `event_order=verify`
- `f32_key_mode=native32` (effective only for F32 input)
- `neighbor_kernel=interior-fast`
- `representative_active_check=recheck`
- `union_kernel=root-carrying`
- `h0_pruning_cache=64k`
- `active_state=separate`
- `interface_state=root-invariant`
- `uf_layout=packed`
- `global_h0_uf_layout=packed`
- for F32 input, `native32` remains 4-byte through H0 disk runs and global birth state

### H2

- `merge_strategy=scan`
- `interface_order=radix`
- `event_order=verify`
- `f32_key_mode=native32` (effective only for F32 input)
- `neighbor_kernel=interior-fast`
- `representative_active_check=recheck`
- `union_kernel=root-carrying`
- `neighbor_root_check=parent-shortcut`
- `active_state=parent-sentinel`
- `interface_state=root-invariant`
- `uf_layout=packed`
- `local_h2_birth_state=compact`
- `phase_trim=before-reduce` on Linux/glibc, otherwise `off`
- `global_h2_birth_state=compact`
- `global_h2_uf_layout=parent-rank`

The local compact birth representation was promoted after exact CX09T1
equivalence and a balanced five-pair real-stack benchmark. CX09T1 was reported
by the instrumented scalar loader as **U16**, so it correctly used the canonical
8-byte `ScalarKey` path; `f32_key_mode=native32` is only representation-relevant
for F32 TIFF stacks. On CX09T1 at slab depth 16, compact local births reduced
median explicit birth storage from 18.33 MiB to 9.16 MiB, total explicit local
H2 state from 22.91 MiB to 13.75 MiB, and median peak RSS from about 93.26 MiB
to 84.33 MiB while preserving all 220,093 intervals and the canonical hash.

The global H2 packed-rank layout remains an ablation/reference option. v1.17
does **not** silently promote it: the frozen global layout is `parent-rank`.

## v1.18-v1.19 H0 F32 promotion

The end-to-end native-F32 H0 path and packed global H0 layout are now frozen production defaults. On real two-slice T1_S26 F32 data, native32 preserved the exact 374,303-interval canonical hash while reducing median wall time from 23.433 s to 19.949 s, peak RSS from about 1013 MiB to 688 MiB, temporary-run bytes by 30.0%, and explicit global H0 state by 30.77%. Packed global H0 then removed the remaining rank byte: explicit global state fell from 246.84 MiB to 219.41 MiB on the same test, all 10 A/B runs retained the same canonical hash, median reduction time improved by about 6.0%, and end-to-end wall time was effectively neutral/slightly better.

For the 3792 x 3792 x 2048 F32 target, slab depth 32 implies about 1.841 billion interface nodes and approximately 13.71 GiB of explicit packed native32 global H0 state.


## Release gate

Run the deterministic suite and, when a representative scalar TIFF stack is
available, the real-stack production equivalence gate:

```bash
scripts/check_production_release.sh /path/to/representative_scalar_stack
```

The stack-backed gate compares no-flag production H0/H2 stream output against
the independent in-memory scalar paths at slab depths 8, 16, and 32 and also
checks the complete reported `PROFILE_CONFIG`.

## Full production profile

After the release gate passes, profile the actual no-flag production command:

```bash
python3 scripts/profile_production_scalar_stream.py \
  /path/to/specimen1 [/path/to/specimen2 ...] \
  --binary target/release/betti_curves \
  --slab-depths 8 16 32 \
  --foreground-connectivity 26 \
  --repeats 3 \
  --output production_scalar_profile.csv
```

The detailed CSV records the binary SHA-256, source pixel type, effective
configuration, wall/user/system time, peak RSS, internal phase timings, interval
count, and canonical persistence hash. The profiler rejects any persistence,
configuration, or source-type inconsistency across repeats/slab depths. A
`_summary.csv` file reports median timing/RSS and the script prints the fastest
slab depth per specimen/mode.

## What remains reference/experimental

The retained alternatives include heap merge, comparison interface ordering,
event resorting, legacy64 F32 keys, generic neighbor traversal, trust-pruner
active checking, conventional unions, H2 full-find same-root checks, alternate
H0 cache sizes/off, alternate active states, vector interface state, local
parent-rank UF, phase-trim off on glibc, tagged local/global H2 births, and the
packed global H2 UF layout.

Further optimization should begin from a fresh v1.19 production profile rather
than from isolated pre-consolidation ablation timings.
