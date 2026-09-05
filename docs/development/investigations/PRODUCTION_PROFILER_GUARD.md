# Production profiler stale-binary guard (v1.13.2)

`profile_production_scalar_stream.py` now validates the effective `PROFILE_CONFIG`
on every recorded run before accepting timing data. It refuses to benchmark a
binary that does not report the consolidated production defaults.

This specifically prevents a common workflow error: source/tests are updated,
but `target/release/betti_curves` is left from an older revision because a
previous release gate stopped before `cargo build --release`.

If the guard fails, rebuild and verify first:

```bash
cargo build --release --locked
python3 validation/check_production_scalar_equivalence.py examples/CX09T1/ \
  --binary target/release/betti_curves --slab-depths 8 16 32
```
