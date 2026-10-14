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

## Optional exact H1 persistence and spatial loops

The ordinary `H1 = H0 + H2 - Euler` calculation provides the **Betti-1 curve**, not the birth/death pairing of individual tunnels. An optional native algebraic Morse backend can also compute the finite **H1 persistence barcode**, preserve unique creator/destroyer pair identifiers, and reconstruct a selected original-grid loop with a chain certificate. The existing H0/H2 and streaming Euler methods remain the default for large volumes.

Install the optional native backend (Python 3.10+, NumPy, Numba and a local C++17 compiler such as `g++`):

```bash
python -m pip install -e '.[morse]'
```

No native binary is bundled. Its C++ code is compiled on first use and cached locally. The backend is experimental; it is neither a generally out-of-core solver nor a promise of tractability for arbitrary 256³ or 512³ intensity fields.

The Morse backend **requires `connectivity=26`**, using the closed-voxel sublevel cubical complex with voxel-minimum face filtration. The 6-connected mode is based on a different cubical model and is not interchangeable with it. Use the same input, exact threshold order, and convention when comparing the independent Betti curve and H1 barcode.

```python
import numpy as np
from betti_curves_py import analyze_volume

volume = np.load('sample_volume.npy', mmap_mode='r')
result = analyze_volume(
    volume,
    connectivity=26,
    h1_backend='morse',
    output_dir='h1_catalog',
    validate_betti=True,
    max_cells=3_000_000,
)

print(result.h1_betti_curve)
print(result.h1.pairs())
print(result.h1.pair_ids)

if len(result.h1):
    certificate = result.h1.reconstruct(result.h1.pair_ids[0])
    print(certificate['verified'], certificate['cycle_cells'])
```

**Memory-conscious H1 catalogs.** The completed native H1 barcode is cached on disk and reused for selective reconstruction. Pair lookup uses a lazily created, sorted creator index, so requesting one representative does not rerun the cohomology pass or materialize a dictionary of all persistence pairs. Reopened catalogs validate input file sizes and modification timestamps before reusing cached results; do not concurrently modify a catalog while reading it. For very large barcodes, set `store_pair_ids=False` in `h1_persistence(...)` or `analyze_volume(..., h1_backend='morse')` to avoid storing one Python string per interval. Use `result.h1.pair_id_at(i)` or `result.h1.iter_pair_ids()` to access exact pair IDs on demand. The default retains the original `pair_ids` tuple for compatibility. This reduces catalog and replay overhead, but does not make the native sparse Morse reducer fully out of core.

For a persistent catalog without recomputing H0, H2, and Euler, call `h1_persistence(volume, connectivity=26, output_dir='h1_catalog')`. The result also offers `at(thresholds)` and `betti_curve()`; interval indices remain associated with their exact pair IDs even when multiple bars share the same birth/death values. A persistent catalog is necessary for reconstructing geometry in a later process. Without `output_dir`, the result owns temporary files that are removed by `result.close()` or when the object is destroyed.

Use the standalone CLI to write `h1_betti_curve.csv` and, in Morse mode, `h1_persistence.csv` and the selective-replay catalog:

```bash
betti-h1-py sample_volume.npy --backend morse --connectivity 26 \
  --max-cells 3000000 --output-dir h1_results
```

For the Betti curve alone, `--backend curve` is the default and supports either connectivity convention. The optional unified API also defaults to `h1_backend='curve'`. A full independent H0/H2/Euler computation can certify a zero Betti-1 curve; if so, `analyze_volume` safely skips the expensive Morse reduction because no positive-length H1 intervals can exist.

**Verification and limits.** With `validate_betti=True`, the Morse barcode-derived Betti-1 curve is checked at *every integer intensity threshold* against the independently calculated H0 + H2 − Euler curve. The closed-voxel convention, finite barcode semantics, and half-open intervals `[birth, death)` are required for that comparison. The algorithm imposes explicit bounds on the full cubical cell count, candidate heap, cancellation log, native reduction work, and replay output. Long-lived or complicated loops can be costly to reconstruct; the reported cycle is an algebraically valid representative, not necessarily a shortest or geometrically unique loop. `torchPersLay` is a separate, optional downstream analysis tool and is not required.

## Bounded native H1 scheduling

The optional Morse backend has an opt-in `indexed_scheduler=True` mode that holds at most one live cancellation candidate per lower cell and respects `max_heap_entries`. It replenishes the bounded candidate window as reduction progresses instead of abandoning simplification when the initial queue is full. Valid same-filtration cancellations preserve persistent homology, even when a queue, time or work budget stops further simplification. Different scheduler choices can give different *representatives and cell IDs* while preserving the birth/death barcode.

Native H1 cohomology writes completed intervals directly to a compact binary catalog (16 bytes per interval), avoiding an in-memory list of all persistence pairs. It has independent limits `cohomology_max_annotation_entries` for live sparse annotation nonzeros and `cohomology_max_output_pairs` for saved intervals. A budget failure never writes completion metadata. **These limits do not bound total process memory**: precursor arrays, native sparse vectors, trace data, and mapped files still need resources. The previous scheduler remains the default; choose the indexed mode explicitly.

```python
from betti_curves_py import h1_persistence
result = h1_persistence(
    "volume.npy", connectivity=26, output_dir="h1_catalog",
    indexed_scheduler=True, max_heap_entries=100_000,
    max_operations=80_000_000,
    cohomology_max_annotation_entries=3_000_000,
    cohomology_max_output_pairs=2_000_000,
    max_cells=20_000_000, store_pair_ids=False,
)
print(len(result), result.statistics)
```

For the CLI, use `--indexed-scheduler`, `--max-heap-entries`, `--max-morse-operations`, `--cohomology-max-annotation-entries`, and `--cohomology-max-output-pairs` with `--backend morse --connectivity 26`. The included benchmark can independently check Betti-1 from 26-connected foreground components, 6-connected enclosed background components, and closed-voxel Euler characteristic:

```bash
python benchmarks/benchmark_h1_scalability.py --size 32 \
  --indexed-scheduler --max-heap-entries 100000 --validate
```

On one synthetic random **128³** 8-level uint8 volume, the complete H1 barcode contained **312,335 intervals**. A minimally reduced run took 25.70 s and peaked at 2,798 MiB; indexed scheduling with a 100,000-entry candidate window reduced 15.57 million post-collapse cells to 1.12 million, taking 35.13 s and peaking at 2,633 MiB. Both barcodes agreed at all eight thresholds with independently computed Betti-1 curves. A geometrically simple 256³ tunnel produced the interval (0,5), with a certified four-edge cycle, in 7.08 s and 1,106 MiB peak RSS. Measurements are individual CPU runs, not general performance guarantees.

**General high-entropy 256³–512³ H1 is not yet solved.** Candidate and annotation limits improve budget control, but the full critical complex and much of the native cohomology state remain in RAM. Further work is needed on external-memory or compressed annotation matrices and cost-aware reduction scheduling.

### Experimental adaptive Morse stopping

An optional marginal-cost rule can stop algebraic Morse cancellations when recent cancellation time exceeds an estimated saving in the later H1 cohomology pass. It is **disabled by default**. Every performed cancellation remains a filtration-compatible chain equivalence, so stopping early changes runtime and representative-cell identifiers, not the persistence barcode.

```python
from betti_curves_py import h1_persistence
result = h1_persistence(
    volume, connectivity=26, output_dir="h1_catalog",
    indexed_scheduler=True,
    adaptive_cohomology_ns_per_cell=800.0,
    adaptive_window=4096, adaptive_patience=2,
)
print(result.statistics["adaptive_stopped"])
```

The rule measures elapsed time in consecutive cancellation windows, each removing two critical generators per pivot, and compares it with `2 * window * adaptive_cohomology_ns_per_cell`. The cost estimate in nanoseconds per critical cell must come from representative prior workloads or a sensitivity experiment. The previous job's observed mean can be calculated as `1e9 * statistics["cohomology_seconds"] / statistics["remaining"]`, but this is **not** a guaranteed marginal saving or an automatically calibrated optimal parameter. Larger estimates allow more work. The default of zero leaves cancellation behavior unchanged. The option requires `indexed_scheduler=True` and retains all resource-budget checks.

Use `betti-h1-py ... --backend morse --indexed-scheduler --adaptive-cohomology-ns-per-cell 800` for the equivalent command-line option. For reproducible comparisons, run `python benchmarks/benchmark_h1_adaptive.py --size 32 --seed 11`, which verifies equality of barcode multisets under minimal, adaptive, and extensive Morse work. On tested random 64³ workloads, the adaptive rule was faster than extensive simplification, but was **not reliably faster than very limited simplification**. It is an experimental scheduling option rather than a general high-entropy large-volume solution.

### Cohomology-aware Morse cost diagnostics

The optional native H1 backend can estimate the marginal saving from cancellations using the **number of surviving critical faces** and **nonzero face-boundary incidences**, not only critical-cell count. It also records observed cohomology annotation updates, XOR work, growth, and peak entries for analysis. These counters do not affect exact persistence results.

```python
from betti_curves_py import h1_persistence

result = h1_persistence(
    volume, connectivity=26, indexed_scheduler=True,
    adaptive_cohomology_ns_per_face=2200.0,
    adaptive_cohomology_ns_per_incidence=0.0,
    adaptive_window=4096, adaptive_patience=2,
    output_dir="h1_catalog",
)
print(result.statistics["critical_faces"])
print(result.statistics["critical_face_incidences"])
print(result.statistics["cohomology_profile"])
```

Both new stopping rates default to zero. The example rate is fitted to a limited synthetic CPU study, **not a universal recommended value**. The face and incidence rates should be calibrated jointly; separately fitted coefficients should not simply be added because those inputs are correlated. The scheduler compares observed cancellation-window time against estimated downstream cohomology saving and stops only after complete, valid filtration-compatible cancellations. Thus it preserves the barcode, but **does not guarantee a faster or globally optimal runtime**. The option requires `indexed_scheduler=True` and is also available as `--adaptive-cohomology-ns-per-face` and `--adaptive-cohomology-ns-per-incidence` in `betti-h1-py`.

To reproduce a bounded synthetic comparison, use `python benchmarks/benchmark_h1_cost_model.py --sizes 24 32 --seeds 2 7` (benchmark-only SciPy dependency). The benchmark validates prospective structural predictions on held-out volumes; observed annotation-work counters are post-hoc and must not be used as an online forecast for the same computation. In 59 individual synthetic runs over 20 volumes, critical-cell count produced held-out R² around 0.907, compared with 0.939 for edges/faces/incidences and 0.965 for **post-hoc** annotation-work features. Face count alone performed approximately as well as the full structural model in this dataset. On the tested 48³–64³ random inputs, a structural stopping rule avoided expensive extensive cancellation but did not reliably beat minimal cancellation or the simpler cell-count rule. The solver is still not fully out of core.

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
