# Benchmark suite

This directory contains the public performance/reproducibility tooling.

- `generate_synthetic_stack.py`: deterministic TIFF fixture generation.
- `run_suite.py`: wall/RSS/I/O benchmark sweeps over mode, slab depth, and thread count.
- `python_naive_baseline.py`: simple SciPy threshold-wise baseline for small volumes.
- `plot_results.py`: scaling plots from benchmark CSV files.
- `criterion/`: Criterion process-level microbenchmarks.
- `data/`: measured development benchmark data used in the README.
- `plots/`: plots generated from `data/published_results.csv`.

Install Python dependencies with:

```bash
python3 -m pip install -r benchmarks/requirements.txt
```

A small generated fixture is committed at `benchmarks/fixtures/synthetic_16/` so reviewers can inspect the expected input layout without generating data first.

The scheduled GitHub benchmark workflow also runs the Rust event-based Betti curves against `python_naive_baseline.py` on the same deterministic U8 fixture and requires the sparse curves to match exactly.
