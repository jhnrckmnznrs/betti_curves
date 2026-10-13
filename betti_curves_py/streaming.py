"""Slab-memory threshold-wise Betti counts for TIFF stacks.

Unlike the global persistence sweep, the peak live state is determined by a
slab and its two z-faces, not all slices. This implementation visits *every*
voxel anew for *each requested threshold*: use a limited threshold list. It
computes exact counts at these thresholds, NOT persistence intervals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

from .io import tiff_files
from .optimized import _jit, _jit_inline, njit


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        from scipy.ndimage import generate_binary_structure
        return generate_binary_structure(3, 1)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    raise ValueError('connectivity must be 6 or 26')


def _load_slab(paths: list[Path], start: int, stop: int,
               shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
    slab = np.empty((stop-start, *shape), dtype=dtype)
    for j, path in enumerate(paths[start:stop]):
        with tifffile.TiffFile(path) as tiff:
            if len(tiff.pages) != 1:
                raise ValueError(f'expected a single-page TIFF: {path}')
            image = tiff.pages[0].asarray()
        if image.shape != shape or image.dtype != dtype:
            raise ValueError(f'inconsistent TIFF slice {path}: {image.shape}, {image.dtype}')
        if dtype.kind == 'f' and not np.isfinite(image).all():
            raise ValueError(f'non-finite values in {path}')
        slab[j] = image
    return slab


@_jit_inline
def _find_frontier(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@_jit
def _merge_packed_numba(keys: np.ndarray, parent: np.ndarray,
                        sizes: np.ndarray, flags: np.ndarray, nprev: int) -> None:
    """Union compact seam edges within compiled code (32-bit packed labels)."""
    for key in keys:
        a = int(key >> np.uint64(32)) - 1
        b = nprev + int(key & np.uint64(0xffffffff)) - 1
        ra = _find_frontier(parent, a)
        rb = _find_frontier(parent, b)
        if ra == rb:
            continue
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        sizes[ra] += sizes[rb]
        flags[ra] = flags[ra] or flags[rb]


@_jit
def _find_many_numba(labels: np.ndarray, nprev: int,
                     parent: np.ndarray) -> np.ndarray:
    roots = np.empty(len(labels), dtype=parent.dtype)
    for i in range(len(labels)):
        roots[i] = _find_frontier(parent, nprev + int(labels[i]) - 1)
    return roots


def _seam_pairs_packed(previous_face: np.ndarray, current_face: np.ndarray,
                       connectivity: int) -> np.ndarray:
    """Compact and globally deduplicate all cross-z component pairs."""
    height, width = previous_face.shape
    directions = [(0, 0)] if connectivity == 6 else [
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ]
    packed = []
    for dy, dx in directions:
        y0 = max(0, -dy)
        y1 = min(height, height-dy)
        x0 = max(0, -dx)
        x1 = min(width, width-dx)
        if y0 >= y1 or x0 >= x1:
            continue
        left = previous_face[y0:y1, x0:x1]
        right = current_face[y0+dy:y1+dy, x0+dx:x1+dx]
        valid = (left != 0) & (right != 0)
        if not np.any(valid):
            continue
        left_ids = left[valid].astype(np.uint64, copy=False)
        right_ids = right[valid].astype(np.uint64, copy=False)
        packed.append((left_ids << np.uint64(32)) | right_ids)
    if not packed:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate(packed))


def _union_python(a: int, b: int, parent: np.ndarray,
                  sizes: np.ndarray, flags: np.ndarray) -> None:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = int(parent[a])
    while parent[b] != b:
        parent[b] = parent[parent[b]]
        b = int(parent[b])
    if a == b:
        return
    if sizes[a] < sizes[b]:
        a, b = b, a
    parent[b] = a
    sizes[a] += sizes[b]
    flags[a] = flags[a] or flags[b]


def _count_at_threshold(
    paths: list[Path], slab_depth: int, threshold: int | float,
    dimension: int, foreground_connectivity: int,
    shape: tuple[int, int], dtype: np.dtype, *, accelerated: bool = True,
) -> int:
    from scipy import ndimage

    h2 = dimension == 2
    conn = (26 if foreground_connectivity == 6 else 6) if h2 else foreground_connectivity
    structure = _structure(conn)
    depth = len(paths)
    height, width = shape
    prev_face = np.zeros((height, width), dtype=np.int32)
    prev_flags = np.zeros(1, dtype=bool)  # 1-based compact frontier labels
    completed = 0

    for z0 in range(0, depth, slab_depth):
        z1 = min(z0+slab_depth, depth)
        slab = _load_slab(paths, z0, z1, shape, dtype)
        mask = slab > threshold if h2 else slab <= threshold
        labels, count = ndimage.label(mask, structure)
        del slab, mask
        nprev = len(prev_flags)-1
        num_nodes = nprev + int(count)
        # 32-bit component labels are sufficient for ordinary in-core slabs;
        # when necessary the Python/scipy parent arrays can use int64.
        index_dtype = np.int32 if num_nodes <= np.iinfo(np.int32).max else np.int64
        parent = np.arange(num_nodes, dtype=index_dtype)
        sizes = np.ones(num_nodes, dtype=index_dtype)
        flags = np.zeros(num_nodes, dtype=bool)
        flags[:nprev] = prev_flags[1:]

        if h2 and count:
            # Every component touching an external x/y face (or a global end
            # z-face) is connected to the single virtual Outside component.
            touches = [labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
                       labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]
            if z0 == 0:
                touches.append(labels[0].ravel())
            if z1 == depth:
                touches.append(labels[-1].ravel())
            boundary_labels = np.unique(np.concatenate(touches))
            boundary_labels = boundary_labels[boundary_labels != 0]
            flags[nprev + boundary_labels - 1] = True

        if nprev and count:
            # scipy.ndimage labels are bounded by signed int32, so each face
            # label fits in the 32-bit half of a packed seam edge.
            keys = _seam_pairs_packed(prev_face, labels[0], conn)
            if accelerated:
                _merge_packed_numba(keys, parent, sizes, flags, nprev)
            else:
                for key in keys:
                    a = int(key >> np.uint64(32)) - 1
                    b = nprev + int(key & np.uint64(0xffffffff)) - 1
                    _union_python(a, b, parent, sizes, flags)

        # Compute compact continuation labels with NumPy unique/index mapping.
        face_labels = np.unique(labels[-1])
        face_labels = face_labels[face_labels != 0]
        if accelerated:
            roots = _find_many_numba(face_labels, nprev, parent)
        else:
            roots = np.empty(len(face_labels), dtype=index_dtype)
            for i, lab in enumerate(face_labels):
                r = nprev + int(lab) - 1
                while parent[r] != r:
                    parent[r] = parent[parent[r]]
                    r = int(parent[r])
                roots[i] = r
        unique_roots, reverse = np.unique(roots, return_inverse=True)
        new_face_map = np.zeros(count + 1, dtype=np.int32)
        new_face_map[face_labels] = reverse.astype(np.int32) + 1
        next_face = new_face_map[labels[-1]]
        active = np.zeros(num_nodes, dtype=bool)
        active[unique_roots] = True
        finished = (parent == np.arange(num_nodes)) & ~active
        if h2:
            finished &= ~flags
        completed += int(np.count_nonzero(finished))
        next_flags = np.zeros(len(unique_roots) + 1, dtype=bool)
        next_flags[1:] = flags[unique_roots]
        prev_face, prev_flags = next_face, next_flags

    completed += (int(np.count_nonzero(~prev_flags[1:])) if h2 else len(prev_flags)-1)
    return completed


def streaming_betti_at(
    stack: str | Path, thresholds: Sequence[int | float],
    *, dimension: int = 0, foreground_connectivity: int = 6,
    slab_depth: int = 8, backend: str = 'auto',
) -> list[tuple[int | float, int]]:
    """Compute exact Betti-0 or Betti-2 *counts* at specified thresholds.

    This mode reads only one z-slab at a time, retaining local label arrays and
    one face of compact component connectivity. Memory is O(slab_depth*H*W),
    not O(total volume); time is proportional to #thresholds times voxel count.

    Output lists every supplied threshold in sorted unique order. For H2,
    background uses the strict inequality `value > threshold`.
    """
    if dimension not in (0, 2):
        raise ValueError('dimension must be 0 or 2')
    if foreground_connectivity not in (6, 26):
        raise ValueError('foreground_connectivity must be 6 or 26')
    if backend not in ('auto', 'numba', 'python'):
        raise ValueError('backend must be auto, numba, or python')
    if backend == 'numba' and njit is None:
        raise ImportError('Numba is not installed: pip install numba')
    accelerated = (njit is not None) if backend == 'auto' else backend == 'numba'
    if slab_depth <= 0:
        raise ValueError('slab_depth must be strictly positive')
    levels = sorted(set(thresholds))
    if not levels:
        raise ValueError('specify at least one threshold')
    if not all(np.isfinite(level) for level in levels):
        raise ValueError('thresholds must be finite')
    paths = tiff_files(stack)
    with tifffile.TiffFile(paths[0]) as tiff:
        if len(tiff.pages) != 1:
            raise ValueError(f'expected single-page TIFF: {paths[0]}')
        image = tiff.pages[0].asarray()
    if image.ndim != 2:
        raise ValueError('expected grayscale 2-D TIFF slices')
    if image.dtype not in (np.dtype('uint8'), np.dtype('uint16'),
                           np.dtype('float32'), np.dtype('float64')):
        raise TypeError(f'unsupported TIFF dtype: {image.dtype}')
    if image.dtype.kind == 'f' and not np.isfinite(image).all():
        raise ValueError('non-finite values in first TIFF slice')
    return [
        (t, _count_at_threshold(paths, slab_depth, t, dimension,
                                foreground_connectivity, image.shape, image.dtype,
                                accelerated=accelerated))
        for t in levels
    ]
