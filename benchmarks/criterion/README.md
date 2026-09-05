# Criterion CLI microbenchmarks

These benchmarks intentionally measure the released CLI as a process. That keeps the production binary as the benchmark subject without forcing internal modules into a public Rust library API solely for benchmarking.

Build the production binary first:

```bash
cargo build --release --locked
cargo bench --manifest-path benchmarks/criterion/Cargo.toml
```

Override the binary or fixture with `BETTI_BENCH_BIN` and `BETTI_BENCH_FIXTURE`.

Criterion measures timing distributions. Use `benchmarks/run_suite.py` for peak RSS and filesystem I/O.
