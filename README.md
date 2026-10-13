# Betti Curves

Betti Curves analyzes the shape of three-dimensional grayscale images. It can count connected regions, tunnels, and enclosed cavities as the intensity threshold changes. This is useful for examining porous materials, bone microstructure, and other volumetric images.

The project provides a Python package and a Rust command-line engine. The Python package supports exact H0 and H2 persistence intervals, Betti curves, and a streaming Euler characteristic curve. Combining matching H0 and H2 curves with the Euler curve gives an H1 **Betti curve** (tunnel counts), not an H1 persistence barcode.

## What the results mean

For a threshold `t`, the foreground includes voxels with intensity at most `t`. As the threshold rises, the foreground grows.

- **H0 / Betti-0** counts disconnected foreground regions.
- **H1 / Betti-1** counts independent tunnels through the foreground.
- **H2 / Betti-2** counts enclosed cavities.
- **Euler characteristic** is `H0 - H1 + H2`. Therefore, `H1 = H0 + H2 - Euler` when all curves use the same filtration and connectivity convention.

You can choose 6-connectivity (voxels connect along faces) or 26-connectivity (voxels also connect across edges and corners). H2 uses the corresponding dual background connectivity. Results from different conventions must not be combined.

## Install the Python package

Python 3.10 or newer is required. From a clone of this repository:

```bash
python -m pip install -e '.[fast,stream]'
```

NumPy and `tifffile` are required. The `fast` option installs Numba for accelerated kernels, and `stream` adds SciPy for threshold-based counting. Run tests with:

```bash
python -m pip install -e '.[fast,stream,test]'
python -m pytest -q
```

## Input images

Use a folder of single-page grayscale TIFF files, one file for each Z slice. All slices must have the same dimensions and pixel type. Files are read in natural numeric order.

```text
sample/
  slice_0000.tif
  slice_0001.tif
  slice_0002.tif
```

The Python persistence engine accepts U8, U16, F32, and F64 data (finite values only). The streaming Euler engine accepts U8 and U16 data. Three-dimensional `.npy` arrays are also supported. Large `.npy` inputs can be memory mapped.

## Compute H0 and H2 in Python

For a small or medium TIFF stack:

```bash
python -m betti_curves_py sample/ \
  --mode both --connectivity 6 --backend numba --output-dir results
```

For larger volumes, use hierarchical or slab processing:

```bash
python -m betti_curves_py sample/ \
  --mode both --connectivity 6 --backend slab \
  --slab-depth 8 --tile-edge 32 \
  --intra-slab boundary-aware \
  --slab-seam-strategy stream26-blocked \
  --output-dir results
```

The program can write H0/H2 persistence pairs and Betti-curve CSV files. The slab and hierarchical approaches can use scratch space, so check disk capacity for large datasets. The source includes other selectable backends for benchmarking and research.

## Compute the Euler curve

The Euler computation reads TIFF slices sequentially without loading the entire volume into memory. Parallel Numba workers maintain separate event histograms to avoid data races.

```bash
python -m betti_curves_py.euler_cli sample/ \
  --connectivity 6 --backend parallel --threads 4 \
  --output-dir euler_results
```

This writes `euler_curve.csv`. To use the serial implementation, select `--backend serial`. If the input is large, start with one run rather than repeated benchmarks.

## Recover the H1 Betti curve

When matching H0 and H2 curves have been computed for the same input and connectivity setting:

```bash
python -m betti_curves_py.euler_cli sample/ \
  --connectivity 6 --backend parallel --threads 4 \
  --h0-curve results/h0_betti_curve.csv \
  --h2-curve results/h2_betti_curve.csv \
  --output-dir euler_results
```

This produces `h1_betti_curve.csv`. It describes **how many** tunnels exist at each threshold. It does not identify individual tunnel birth and death pairs. The library also supports sampled-threshold H1 counts if full H0/H2 persistence is too costly.

## Python API

```python
from betti_curves_py import euler_curve, persistence

# volume can be a 3-D NumPy array
curve = euler_curve(volume, connectivity=6, backend="parallel", threads=4)
print(curve.at(100))

h0 = persistence(volume, dimension=0, foreground_connectivity=6)
```

## Rust command-line engine

The Rust engine can compute exact H0/H2 persistence and Betti curves on TIFF stacks using streaming and hierarchical methods. Build it with:

```bash
cargo build --release --locked
```

For available commands and parameters:

```bash
target/release/betti_curves --help
```

The Rust and Python engines are separate implementations. Their memory and runtime characteristics depend on the image and processing mode. Validate results on a representative crop before analyzing an entire scan.

## Performance and limitations

The Euler algorithm uses two adjacent Z planes and integer event counters. Its memory requirements therefore depend primarily on a slice's area, the histogram, and parallel-worker count. The exact H0/H2 persistence algorithms may require additional boundary summaries, temporary arrays, and disk space. The package has been tested on synthetic volumes; processing a complete 2,048 × 3,700 × 3,700 image stack is not yet a demonstrated benchmark.

All thresholds use the sublevel rule (intensity less than or equal to the threshold). Persistence intervals use the half-open `[birth, death)` convention. The Euler identity requires compatible cubical interpretations, so do not mix curves produced using different settings.

## License

MIT. See [LICENSE](LICENSE).
