# Zeroth and Second Betti Curves from TIFF Stacks

A Rust implementation for computing **Betti-0** and **Betti-2** curves from large 3D grayscale TIFF stacks.

The project includes two slabwise approaches:

- a **threshold-wise baseline**, which computes each threshold independently;
- an **event-based algorithm**, which processes each slab once in filtration order and is substantially faster when the image contains many distinct intensity values.

Both approaches use z-slab decomposition and union-find so the entire volume does not need to be stored in memory at once.

## What this computes

Given a grayscale 3D image stack `I(x, y, z)`, the program uses the foreground sublevel filtration

```text
X_t = { (x, y, z) : I(x, y, z) <= t }
```

and computes sparse step-function CSVs for:

- `beta0(t)`: the number of foreground connected components;
- `beta2(t)`: the number of enclosed background components, using dual background connectivity and outside-boundary tracking.

The background at foreground threshold `t` is

```text
B_t = { (x, y, z) : I(x, y, z) > t }
```

A background component contributes to Betti-2 only if it is not connected to the outside of the image domain.

## Connectivity convention

Foreground connectivity is chosen by the user:

```text
6   = face adjacency
26  = face + edge + corner adjacency
```

For Betti-2, the background connectivity is chosen as the dual:

```text
foreground 6  -> background 26
foreground 26 -> background 6
```

Background components that touch the global image boundary are treated as connected to the outside and are not counted as voids.

## Input format

The input is a directory of 2D grayscale TIFF slices:

```text
slices/
  slice_0000.tif
  slice_0001.tif
  slice_0002.tif
  ...
```

Filenames are sorted lexicographically, so zero-padded names should be used.

Supported pixel types:

- grayscale `u8`;
- grayscale `u16`.

`u8` values are promoted to `u16` internally.

## Installation

Install Rust, clone the repository, and build the release executable:

```bash
git clone https://github.com/jhnrckmnznrs/betti_curves.git
cd betti_curves
cargo build --release
```

For development checks:

```bash
cargo fmt
cargo clippy --release -- -D warnings
cargo test
```

## Usage

```bash
cargo run --release -- <tiff_directory> [slab_depth] [foreground_connectivity] [mode]
```

Available modes:

```text
betti0         threshold-wise Betti-0
betti2         threshold-wise Betti-2
both           threshold-wise Betti-0 and Betti-2
event-betti0   event-based slabwise Betti-0
event-betti2   event-based slabwise Betti-2
```

Examples:

```bash
cargo run --release -- ./examples/known_betti0_tiff_stack/slices 1 6 betti0
cargo run --release -- ./examples/known_betti0_tiff_stack/slices 2 6 event-betti0

cargo run --release -- ./slices 4 6 betti2
cargo run --release -- ./slices 4 6 event-betti2

cargo run --release -- ./slices 4 6 both
cargo run --release -- ./slices 4 26 event-betti0
```

Use Rayon thread control when benchmarking:

```bash
RAYON_NUM_THREADS=4 cargo run --release -- ./slices 4 6 event-betti0
RAYON_NUM_THREADS=4 cargo run --release -- ./slices 4 6 event-betti2
```

## Output format

The program writes sparse CSV files that record only thresholds where the Betti number changes.

Threshold-wise output:

```text
global_betti0_curve_changes.csv
global_betti2_curve_changes.csv
```

Event-based output:

```text
event_global_betti0_curve_changes.csv
event_global_betti2_curve_changes.csv
```

Each row means that, starting at the listed threshold, the Betti number has the listed value until the next threshold row.

Example:

```csv
threshold,betti0
0,0
10,1
20,2
30,3
40,2
60,1
```

## Algorithms

### Threshold-wise Betti-0

For each threshold `t`:

1. Read the volume one z-slab at a time.
2. Activate voxels with `value <= t`.
3. Compute connected components inside each slab using union-find.
4. Store component labels on the first and last z-faces.
5. Reconcile adjacent slabs through their z-faces.
6. Record the global Betti-0 value.

### Threshold-wise Betti-2

For each foreground threshold `t`, the program computes connected components of

```text
B_t = { I > t }
```

using the dual background connectivity.

The algorithm tracks whether each background component touches the global image boundary. Components connected to the outside are excluded from Betti-2.

### Event-based Betti-0

The event-based Betti-0 algorithm avoids recomputing the volume at every threshold.

For each slab:

1. Activate voxels in increasing intensity order.
2. Create a new component for each activated voxel.
3. Union the voxel with all already-active foreground neighbors.
4. Record local Betti-0 changes.
5. Record interface merge events describing when boundary components become connected inside the slab.
6. Store interface-node information on the lower and upper z-faces.

After all slabs are processed, a final global process combines local events, interface events, and cross-slab adjacencies to obtain the full-volume Betti-0 curve.

### Event-based Betti-2

The event-based Betti-2 algorithm processes the background superlevel filtration.

For each slab:

1. Activate background voxels in decreasing intensity order.
2. Track background connected components using the dual connectivity.
3. Track whether each component is connected to the outside.
4. Record local changes in the number of non-outside background components.
5. Record interface merge events and outside-connectivity events.
6. Store interface-node information on the lower and upper z-faces.

A final global process combines these slab summaries and produces the Betti-2 curve in foreground threshold order.

## Parallelism

The two approaches use different forms of parallelism.

### Threshold-wise parallelism

Each threshold is independent, so the threshold-wise algorithm parallelizes across thresholds.

This is simple, but each worker rereads and reprocesses the TIFF stack for its assigned thresholds.

### Event-based parallelism

The filtration order is sequential, so event-based computation does not parallelize naturally across thresholds.

Instead, slabs are processed independently in parallel. Their event summaries are then combined by a global reducer.

Control the number of slab workers with:

```bash
RAYON_NUM_THREADS=<threads>
```

## Performance notes

For the threshold-wise algorithm, runtime scales approximately as:

```text
number of unique thresholds × number of voxels
```

Random 16-bit images often contain many distinct intensity values, so the threshold-wise algorithm can be very slow.

The event-based algorithm processes each voxel and its local neighbors once per slab sweep. Its dominant work is close to linear in the number of voxels, up to union-find and interface-reconciliation costs.

Larger slab depths reduce the number of slab interfaces but increase memory use per worker. More Rayon threads may reduce runtime, but several slabs can be resident in memory simultaneously.

Benchmark different slab depths and thread counts:

```bash
RAYON_NUM_THREADS=1 cargo run --release -- ./slices 25 6 event-betti0
RAYON_NUM_THREADS=4 cargo run --release -- ./slices 50 6 event-betti0
RAYON_NUM_THREADS=8 cargo run --release -- ./slices 100 6 event-betti0
```

On Linux, runtime and peak memory can be measured with:

```bash
RAYON_NUM_THREADS=4 /usr/bin/time -v \
cargo run --release -- ./slices 100 26 event-betti0
```

The main reported quantities are:

```text
Elapsed (wall clock) time
Maximum resident set size
```

## Validation

The threshold-wise implementation is intended to serve as a correctness baseline for the event-based implementation.

For a test image, compare:

```bash
cargo run --release -- ./slices 1 6 betti0
cargo run --release -- ./slices 1 6 event-betti0
```

and:

```bash
cargo run --release -- ./slices 1 6 betti2
cargo run --release -- ./slices 1 6 event-betti2
```

The Betti curves should agree.

The output should also be invariant under valid choices of slab depth:

```bash
cargo run --release -- ./slices 1 6 event-betti0
cargo run --release -- ./slices 2 6 event-betti0
cargo run --release -- ./slices 4 6 event-betti0
```

## Limitations

- The decomposition is based on z-slabs rather than arbitrary cropped 3D blocks.
- TIFF files are sorted lexicographically and therefore require zero-padded filenames.
- TIFF slices are decoded during slab reads; chunked formats such as Zarr, N5, or HDF5 may be more efficient for some workflows.
- Increasing the number of Rayon workers can substantially increase peak memory usage.

## Repository

Source code:

```text
https://github.com/jhnrckmnznrs/betti_curves
```

