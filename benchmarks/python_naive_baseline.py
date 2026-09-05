#!/usr/bin/env python3
"""Simple threshold-wise SciPy baseline for small TIFF stacks.

This is intentionally not optimized. It exists to provide an independent,
readable baseline for Betti curves and to make the cost of rescanning every
threshold visible in benchmarks.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy import ndimage


def natural_key(path: Path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def load_stack(root: Path) -> np.ndarray:
    paths = sorted([*root.glob("*.tif"), *root.glob("*.tiff")], key=natural_key)
    if not paths:
        raise SystemExit(f"no TIFF slices found in {root}")
    planes = [np.asarray(Image.open(path)) for path in paths]
    shape = planes[0].shape
    if any(plane.shape != shape for plane in planes):
        raise SystemExit("mixed slice dimensions")
    return np.stack(planes, axis=0)


def structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError(connectivity)


def boundary_labels(labels: np.ndarray) -> set[int]:
    faces = [labels[0], labels[-1], labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1]]
    result = set(int(v) for face in faces for v in np.unique(face))
    result.discard(0)
    return result


def sparse_changes(rows: list[tuple[float, int]]) -> list[tuple[float, int]]:
    out = []
    previous = None
    for threshold, value in rows:
        if value != previous:
            out.append((threshold, value))
            previous = value
    return out


def write_curve(path: Path, header: str, rows: list[tuple[float, int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", header])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stack", type=Path)
    parser.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-thresholds", type=int, default=4096, help="refuse larger unique-value sets by default")
    args = parser.parse_args()

    volume = load_stack(args.stack)
    thresholds = np.unique(volume)
    if len(thresholds) > args.max_thresholds:
        raise SystemExit(
            f"naive baseline would scan {len(thresholds)} thresholds; "
            f"increase --max-thresholds explicitly if this is intentional"
        )

    fg_structure = structure(args.foreground_connectivity)
    bg_structure = structure(26 if args.foreground_connectivity == 6 else 6)
    h0_rows = []
    h2_rows = []
    for threshold in thresholds:
        _, h0 = ndimage.label(volume <= threshold, structure=fg_structure)
        labels, bg_count = ndimage.label(volume > threshold, structure=bg_structure)
        outside = boundary_labels(labels)
        h2 = int(bg_count) - len(outside)
        h0_rows.append((float(threshold), int(h0)))
        h2_rows.append((float(threshold), h2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_curve(args.output_dir / "python_betti0.csv", "betti0", sparse_changes(h0_rows))
    write_curve(args.output_dir / "python_betti2.csv", "betti2", sparse_changes(h2_rows))
    print(f"shape={volume.shape[::-1]} thresholds={len(thresholds)}")


if __name__ == "__main__":
    main()
