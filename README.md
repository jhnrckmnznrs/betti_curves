# Betti Curves from TIFF Stacks

A Rust implementation for computing **Betti-0** and **Betti-2** curves from large 3D grayscale TIFF stacks.

This implementation is intentionally conservative: it computes each threshold independently, using z-slab decomposition and union-find. 

## What this computes

Given a grayscale 3D image stack `I(x, y, z)`, this program uses the foreground sublevel filtration

```text
X_t = { (x, y, z) : I(x, y, z) <= t }
```

and computes sparse step-function CSVs for:

- `beta0(t)`: number of foreground connected components.
- `beta2(t)`: number of enclosed background components, using the dual background connectivity and outside-boundary tracking.

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

The Betti-2 computation tracks whether each background component touches the global image boundary. Components that touch the boundary are treated as connected to the outside and are not counted as voids.

## Input format

The input is a directory of 2D grayscale TIFF slices:

```text
slices/
  slice_0000.tif
  slice_0001.tif
  slice_0002.tif
  ...
```

Filenames are sorted lexicographically, so use zero-padded names.

Supported pixel types:

- grayscale `u8`
- grayscale `u16`

`u8` values are promoted to `u16` internally.

## Installation

Install Rust, then clone this repository and run:

```bash
cargo build --release
```

## Usage

```bash
cargo run --release -- <tiff_directory> [slab_depth] [foreground_connectivity] [mode]
```

Examples:

```bash
cargo run --release -- ./examples/known_betti0_tiff_stack/slices 1 6 betti0
cargo run --release -- ./examples/known_betti0_tiff_stack/slices 2 6 betti0
cargo run --release -- ./slices 4 6 betti2
cargo run --release -- ./slices 4 6 both
cargo run --release -- ./slices 4 26 both
```

Use Rayon thread control when benchmarking:

```bash
RAYON_NUM_THREADS=4 cargo run --release -- ./slices 4 6 both
```

## Output format

The program writes sparse CSV files that record only thresholds where the Betti number changes.

```text
global_betti0_curve_changes.csv
global_betti2_curve_changes.csv
```

Each row means: starting at this threshold, the Betti number has the listed value until the next threshold row.

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

## Algorithm

### Betti-0

For each threshold `t`:

1. Read the volume one z-slab at a time.
2. Activate voxels with `value <= t`.
3. Compute connected components inside the slab using union-find.
4. Store labels on the first and last z-faces.
5. Reconcile adjacent slabs through their z-faces.
6. Record the global Betti-0 value.

### Betti-2

For foreground threshold `t`, the background is

```text
B_t = { I > t }
```

The program computes background connected components using the dual connectivity. A component is counted toward Betti-2 only if it does **not** touch the global image boundary.

## Performance notes

Runtime scales approximately as:

```text
number of unique thresholds × number of voxels
```

The implementation parallelizes over thresholds with Rayon. It also uses z-slab decomposition so the full volume does not have to be held in memory at once.

For large datasets, benchmark different slab depths:

```bash
RAYON_NUM_THREADS=4 cargo run --release -- ./slices 1 6 betti0
RAYON_NUM_THREADS=4 cargo run --release -- ./slices 2 6 betti0
RAYON_NUM_THREADS=4 cargo run --release -- ./slices 4 6 betti0
```

## Limitations

- The implementation recomputes connected components for each unique threshold.
- TIFF files are decoded repeatedly across thresholds.
- The decomposition is z-slab based, not arbitrary 3D cropped blocks.
- The file ordering is lexicographic and assumes zero-padded filenames
