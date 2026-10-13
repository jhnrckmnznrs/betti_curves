"""Numba-accelerated persistence kernels and stable counting sort.

These optimize the original global union-find sweep; they do not implement
slab-bounded *persistence*. The 32-bit sweep is only called when N + 1 fits
signed int32. Original kernels remain in core.py as reference implementations.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _jit_inline(function):
    return njit(cache=True, inline='always')(function) if njit is not None else function


def _jit(function):
    return njit(cache=True)(function) if njit is not None else function


@_jit_inline
def _find_fast(parent, p):
    root = p
    while parent[root] != root:
        root = parent[root]
    while parent[p] != p:
        nxt = parent[p]
        parent[p] = root
        p = nxt
    return root


@_jit_inline
def _join_roots(ra, rb, level, values, parent, sizes, elder,
                h2, outside, births, deaths, children, ancestors, count, tree):
    """Root arguments are already resolved; return the surviving *UF root*."""
    ia, ib = elder[ra], elder[rb]
    if h2 and (ia == outside or ib == outside):
        older_a = ia == outside
    elif h2:
        older_a = values[ia] > values[ib] or (values[ia] == values[ib] and ia < ib)
    else:
        older_a = values[ia] < values[ib] or (values[ia] == values[ib] and ia < ib)
    old = ia if older_a else ib
    young = ib if older_a else ia
    if values[young] != level:
        # The caller has already ensured space for one more interval.
        if h2:
            births[count] = level
            deaths[count] = values[young]
        else:
            births[count] = values[young]
            deaths[count] = level
        if tree:
            children[count] = young
            ancestors[count] = old
        count += 1
    if sizes[ra] < sizes[rb]:
        parent[ra] = rb
        sizes[rb] += sizes[ra]
        elder[rb] = old
        return rb, count
    parent[rb] = ra
    sizes[ra] += sizes[rb]
    elder[ra] = old
    return ra, count


@_jit_inline
def _join_with_growing_buffers(
    ra, rb, level, values, parent, sizes, elder, h2, outside,
    births, deaths, children, ancestors, count, tree, n,
):
    """Amortized O(1) append, allocating only for emitted intervals."""
    if count == len(births):
        capacity = min(n, max(count + 1, 2 * len(births)))
        next_births = np.empty(capacity, dtype=values.dtype)
        next_deaths = np.empty(capacity, dtype=values.dtype)
        next_births[:count] = births[:count]
        next_deaths[:count] = deaths[:count]
        births, deaths = next_births, next_deaths
        if tree:
            next_children = np.empty(capacity, dtype=np.int32)
            next_ancestors = np.empty(capacity, dtype=np.int32)
            next_children[:count] = children[:count]
            next_ancestors[:count] = ancestors[:count]
            children, ancestors = next_children, next_ancestors
    ra, count = _join_roots(ra, rb, level, values, parent, sizes, elder,
                            h2, outside, births, deaths, children, ancestors,
                            count, tree)
    return ra, count, births, deaths, children, ancestors


@_jit
def sweep_fast(values, order, depth, height, width, offsets, h2, tree):
    """Exact root-cached sweep, int32 union-find, fast interior traversal.

    Must only be called for N <= int32.max-1; outer API checks this limit.
    Arrays of intervals grow geometrically instead of preallocating N slots.
    """
    n = len(values)
    outside = n
    m = n + int(h2)
    parent = np.full(m, -1, dtype=np.int32)
    sizes = np.ones(m, dtype=np.int32)
    elder = np.empty(m, dtype=np.int32)
    capacity = min(n, 4096)
    births = np.empty(capacity, dtype=values.dtype)
    deaths = np.empty(capacity, dtype=values.dtype)
    children = np.empty(capacity if tree else 0, dtype=np.int32)
    ancestors = np.empty(capacity if tree else 0, dtype=np.int32)
    if h2:
        parent[outside] = outside
        elder[outside] = outside
    plane = height * width
    linear_offsets = np.empty(len(offsets), dtype=np.int64)
    for k in range(len(offsets)):
        linear_offsets[k] = int(offsets[k, 0]) * plane + int(offsets[k, 1]) * width + int(offsets[k, 2])
    count = 0
    for pp in order:
        p = int(pp)
        val = values[p]
        z = p // plane
        y = (p % plane) // width
        x = p % width
        parent[p] = p
        elder[p] = p
        root = p
        interior = (0 < z < depth-1 and 0 < y < height-1 and 0 < x < width-1)
        if h2 and not interior:
            # Only a true volume boundary joins Outside, not any "not interior"
            # condition arising from nonboundary one-cell spatial dimensions.
            if z == 0 or z == depth - 1 or y == 0 or y == height - 1 or x == 0 or x == width - 1:
                other = _find_fast(parent, outside)
                if root != other:
                    root, count, births, deaths, children, ancestors = _join_with_growing_buffers(
                        root, other, val, values, parent, sizes, elder, h2, outside,
                        births, deaths, children, ancestors, count, tree, n)
        if interior:
            for k in range(len(offsets)):
                q = p + linear_offsets[k]
                if parent[q] != -1:
                    other = _find_fast(parent, q)
                    if root != other:
                        root, count, births, deaths, children, ancestors = _join_with_growing_buffers(
                            root, other, val, values, parent, sizes, elder, h2, outside,
                            births, deaths, children, ancestors, count, tree, n)
        else:
            for k in range(len(offsets)):
                nz = z + int(offsets[k, 0])
                ny = y + int(offsets[k, 1])
                nx = x + int(offsets[k, 2])
                if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                    q = nz * plane + ny * width + nx
                    if parent[q] != -1:
                        other = _find_fast(parent, q)
                        if root != other:
                            root, count, births, deaths, children, ancestors = _join_with_growing_buffers(
                                root, other, val, values, parent, sizes, elder, h2, outside,
                                births, deaths, children, ancestors, count, tree, n)
    if not h2:
        root = _find_fast(parent, 0)
        if count == len(births):
            nb = np.empty(count+1, dtype=values.dtype)
            nd = np.empty(count+1, dtype=values.dtype)
            nb[:count] = births[:count]
            nd[:count] = deaths[:count]
            births, deaths = nb, nd
            if tree:
                nc = np.empty(count+1, dtype=np.int32)
                na = np.empty(count+1, dtype=np.int32)
                nc[:count] = children[:count]
                na[:count] = ancestors[:count]
                children, ancestors = nc, na
        births[count] = values[elder[root]]
        deaths[count] = 0  # essential-H0 marker; not a finite death
        if tree:
            children[count] = elder[root]
            ancestors[count] = -1
        count += 1
    return (births[:count].copy(), deaths[:count].copy(),
            children[:count].copy(), ancestors[:count].copy())


@_jit
def _counting_fill(values, descending, output, bins):
    """Stable integer ordering with O(N+B) work, B=256 or 65536."""
    n = len(values)
    counts = np.zeros(bins, dtype=np.int64)
    for v in values:
        counts[int(v)] += 1
    total = 0
    if descending:
        for k in range(bins-1, -1, -1):
            cnt = counts[k]
            counts[k] = total
            total += cnt
    else:
        for k in range(bins):
            cnt = counts[k]
            counts[k] = total
            total += cnt
    for i in range(n):
        k = int(values[i])
        idx = counts[k]
        output[idx] = i
        counts[k] = idx+1


def stable_voxel_order(values: np.ndarray, h2: bool, *, counting: bool = True) -> np.ndarray:
    """Sort by intensity, retaining increasing voxel-index ties.

    Counting sort is used for sufficiently large U8/U16 arrays when Numba is
    installed; NumPy stable argsort is the safe fallback for all accepted dtypes.
    """
    if counting and njit is not None and values.dtype.kind == 'u' and (
        len(values) >= (4096 if values.dtype.itemsize == 1 else 65536)
    ):
        index_dtype = np.int32 if len(values) <= np.iinfo(np.int32).max - 1 else np.int64
        output = np.empty(len(values), dtype=index_dtype)
        _counting_fill(values, h2, output, 256 if values.dtype.itemsize == 1 else 65536)
        return output
    if h2:
        if values.dtype.kind == 'u':
            return np.argsort(np.bitwise_not(values), kind='stable')
        return np.argsort(-values, kind='stable')
    return np.argsort(values, kind='stable')
