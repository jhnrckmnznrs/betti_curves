# Benchmarks and reproducibility

## Published measurements

The repository includes a small set of development measurements under `benchmarks/data/` because they demonstrate the engineering properties that motivated the current architecture.

The public README table is generated from `benchmarks/data/published_results.csv`.

### Provenance

- `cx09t1_h0_attach_pruning.csv`: H0 hierarchy / attach-pruning ablation on CX09T1.
- `cx09t1_h2_cross_storage_summary.csv`: H2 direct-cross ablation.
- `cx09t1_h2_outside_structural_pruning_summary.csv`: H2 outside-structural-pruning ablation.
- `t1_s26_h0_real_prefix.csv`: 64- and 128-slice prefixes of a real 3792×3792 F32 synchrotron stack.

Historical runs did not record `RAYON_NUM_THREADS`. They are therefore marked `not recorded`; the benchmark runner added for the public repository always records the requested thread count.

## What the benchmark runner records

`benchmarks/run_suite.py` records:

- input shape and voxel count;
- mode and slab depth;
- foreground connectivity;
- Rayon thread count;
- wall/user/system time;
- peak RSS;
- major/minor faults;
- filesystem inputs/outputs;
- output path and interval/curve row count;
- executable SHA-256;
- command line.

On Linux, `/usr/bin/time -v` is used for memory and filesystem counters.

## Deterministic synthetic data

```bash
python3 benchmarks/generate_synthetic_stack.py \
  --output /tmp/tda-stack \
  --shape 128 128 128 \
  --pattern blobs \
  --seed 42
```

Patterns are deterministic for a fixed seed and include `noise`, `blobs`, `layers`, and `shell`.

## Run a scaling sweep

```bash
python3 benchmarks/run_suite.py /tmp/tda-stack \
  --binary target/release/betti_curves \
  --modes h0-scalar-stream h2-scalar-stream \
  --slab-depths 8 16 32 \
  --threads 1 2 4 8 \
  --repeats 3 \
  --output benchmark_results.csv
```

## Generate plots

```bash
python3 benchmarks/plot_results.py benchmark_results.csv --output-dir benchmark_plots
```

When the CSV contains numeric thread counts, the plotter generates `speedup_vs_threads.svg` automatically. It never invents missing thread counts for historical data.

## Naive Python baseline

`benchmarks/python_naive_baseline.py` intentionally uses a straightforward threshold-wise SciPy connected-component calculation. It is suitable only for small benchmark volumes, but it makes the algorithmic gap concrete and provides an implementation outside Rust for curve-level cross-checking.

```bash
python3 benchmarks/python_naive_baseline.py /tmp/tda-stack \
  --foreground-connectivity 26 \
  --output-dir /tmp/python-baseline
```

## Interpretation

Do not compare absolute timings across different machines as though they were one controlled experiment. Use the committed historical results for architecture-scale evidence and use the benchmark suite to make before/after claims on the same host.

For performance pull requests, report medians over at least three runs on small/medium fixtures and one scale test that exercises the relevant memory/I/O path.
