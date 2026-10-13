"""TIFF-stack ingestion and robust CSV output."""
from __future__ import annotations

import csv
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .core import Intervals, _scalar, betti_curve


def natural_key(path: Path):
    return [int(chunk) if chunk.isdecimal() else chunk.lower()
            for chunk in re.split(r'(\d+)', path.name)]


def tiff_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted((p for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in ('.tif', '.tiff')),
                   key=natural_key)
    if not paths:
        raise ValueError(f'no .tif/.tiff image slices in {root}')
    return paths


def read_tiff_stack(directory: str | Path, *, mmap_path: str | Path | None = None) -> np.ndarray:
    """Read naturally sorted single-page grayscale TIFFs as a (Z,Y,X) array.

    Optional mmap_path writes a raw binary disk-backed array, which remains on
    disk after this function returns. It does not bound the memory needed for
    subsequent persistence sorting / global union-find.
    """
    paths = tiff_files(directory)
    with tifffile.TiffFile(paths[0]) as tiff:
        if len(tiff.pages) != 1:
            raise ValueError(f'expect exactly one page per TIFF: {paths[0]}')
        first = tiff.pages[0].asarray()
    if first.ndim != 2:
        raise ValueError(f'TIFF must be grayscale 2D: {paths[0]}')
    if first.dtype not in (np.dtype('uint8'), np.dtype('uint16'),
                           np.dtype('float32'), np.dtype('float64')):
        raise TypeError(f'unsupported TIFF dtype {first.dtype}')
    shape = (len(paths), *first.shape)
    if mmap_path is not None:
        dest = Path(mmap_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        volume = np.memmap(dest, mode='w+', dtype=first.dtype, shape=shape)
    else:
        volume = np.empty(shape, dtype=first.dtype)
    for z, path in enumerate(paths):
        if z == 0:
            image = first
        else:
            with tifffile.TiffFile(path) as tiff:
                if len(tiff.pages) != 1:
                    raise ValueError(f'expect exactly one page per TIFF: {path}')
                image = tiff.pages[0].asarray()
        if image.ndim != 2 or image.shape != first.shape or image.dtype != first.dtype:
            raise ValueError(f'inconsistent shape/dtype of TIFF slice {path}: '
                             f'{image.shape}/{image.dtype}, expected {first.shape}/{first.dtype}')
        if image.dtype.kind == 'f' and not np.isfinite(image).all():
            raise ValueError(f'non-finite scalar values in {path}')
        volume[z, :, :] = image
    if isinstance(volume, np.memmap):
        volume.flush()
    return volume


@contextmanager
def atomic_csv(path: str | Path, header: list[str]) -> Iterator[csv.writer]:
    """Write to a same-directory temporary file and atomically replace."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(header)
            yield writer
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def write_persistence(path: str | Path, intervals: Intervals) -> None:
    with atomic_csv(path, ['birth', 'death']) as writer:
        for i in range(len(intervals)):
            writer.writerow([_scalar(intervals.births[i]),
                             'inf' if intervals.essential and i == len(intervals) - 1
                             else _scalar(intervals.deaths[i])])


def write_betti_curve(path: str | Path, intervals: Intervals) -> None:
    header = 'betti0' if intervals.dimension == 0 else 'betti2'
    with atomic_csv(path, ['threshold', header]) as writer:
        writer.writerows(betti_curve(intervals))


def _branch_rows(intervals: Intervals, *, canonicalize: bool):
    """Construct positive-branch elder trees; no zero-length branches retained."""
    if intervals.branch_ids is None or intervals.parent_branch_ids is None:
        raise ValueError('compute persistence with branch_tree=True first')
    h2 = intervals.dimension == 2
    branches = [int(i) for i in intervals.branch_ids]
    if len(set(branches)) != len(branches):
        raise AssertionError('branch recorded more than once')
    parent_by_id = dict(zip(branches, map(int, intervals.parent_branch_ids)))
    bounds_by_id = {
        branch: (_scalar(intervals.births[i]),
                 'inf' if intervals.essential and i == len(intervals) - 1
                 else _scalar(intervals.deaths[i]))
        for i, branch in enumerate(branches)
    }
    if h2:
        # The explicit Outside branch is listed first as in the Rust exporter.
        outside = next((parent for parent in parent_by_id.values()
                        if parent not in parent_by_id), None)
        if outside is None:
            # A volume without positive H2 intervals still has an Outside root.
            if branches:
                raise AssertionError('H2 background tree lost its outside branch')
            outside = -2
        ordered = [outside] + sorted(branches)
        bounds_by_id[outside] = ('-inf', 'inf')
        parent_by_id[outside] = -1
    else:
        ordered = sorted(branches)
    dense = {key: ix for ix, key in enumerate(ordered)}
    plateau_col = 0 if h2 else 1
    plateau = {key: bounds_by_id[key][plateau_col] for key in ordered}
    # Path-compress equal-plateau parent chains once, rather than repeatedly
    # traversing the same chains for each child. This remains iterative even
    # on very deep trees (avoids Python recursion limits).
    skip_cache: dict[int, int] = {}

    def after_plateau(node: int) -> int:
        origin = node
        path: list[int] = []
        seen: set[int] = set()
        while node != -1 and node not in skip_cache:
            if node in seen:
                raise AssertionError('branch parent cycle')
            seen.add(node)
            path.append(node)
            parent = parent_by_id[node]
            if parent == -1 or plateau[parent] != plateau[node]:
                result = parent
                break
            node = parent
        else:
            result = -1 if node == -1 else skip_cache[node]
        for ident in path:
            skip_cache[ident] = result
        return skip_cache[origin]

    nodes = []
    for ident in ordered:
        parent = parent_by_id[ident]
        if canonicalize and parent != -1 and plateau[parent] == plateau[ident]:
            parent = after_plateau(parent)
        b, d = bounds_by_id[ident]
        nodes.append((dense[ident], '' if parent == -1 else dense[parent], b, d))
    return nodes


def write_branch_tree(prefix: str | Path, intervals: Intervals, *, canonicalize: bool = True) -> tuple[Path, Path]:
    """Write Rust-style node and edge CSVs (positive branches only).

    Branch IDs use a deterministic voxel-index tie break. Parent links are
    plateau-canonical by default, but can differ from Rust's implementation
    in datasets with equal-value voxels, where the tree is not unique.
    """
    prefix = Path(prefix)
    nodes_path = prefix.with_name(prefix.name + '_nodes.csv')
    edges_path = prefix.with_name(prefix.name + '_edges.csv')
    nodes = _branch_rows(intervals, canonicalize=canonicalize)
    with atomic_csv(nodes_path, ['node', 'parent', 'birth_value', 'death_value']) as w:
        w.writerows(nodes)
    by_id = {row[0]: row for row in nodes}
    with atomic_csv(edges_path, ['from_node', 'to_node', 'from_birth_value',
                                 'from_death_value', 'to_birth_value', 'to_death_value']) as w:
        for i, parent, b, d in nodes:
            if parent != '':
                p = by_id[parent]
                w.writerow([i, parent, b, d, p[2], p[3]])
    return nodes_path, edges_path
