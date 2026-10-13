"""Exact H0/H2 persistence on 3-D scalar voxel volumes.

Sublevel foreground X_t = {v: volume[v] <= t}; dual background uses
{v: volume[v] > t}. 6-foreground/26-background or the reverse.

This is a global union-find implementation, *not* the slab-bounded
hierarchical implementation of the original Rust repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .optimized import sweep_fast, stable_voxel_order

try:
    from numba import njit
except ImportError:
    njit = None


def _jit(func):
    return njit(cache=True)(func) if njit is not None else func


@_jit
def _find(parent: np.ndarray, index: int) -> int:
    root = index
    while parent[root] != root:
        root = parent[root]
    while parent[index] != index:
        next_index = parent[index]
        parent[index] = root
        index = next_index
    return root


@_jit
def _union(
    a: int, b: int, level, values: np.ndarray, parent: np.ndarray,
    size: np.ndarray, elder: np.ndarray, h2: bool, outside: int,
    births: np.ndarray, deaths: np.ndarray,
    children: np.ndarray, ancestors: np.ndarray, count: int, tree: bool,
) -> int:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra == rb:
        return count
    ia = elder[ra]
    ib = elder[rb]
    if h2 and (ia == outside or ib == outside):
        older_a = ia == outside
    elif h2:
        older_a = values[ia] > values[ib] or (values[ia] == values[ib] and ia < ib)
    else:
        older_a = values[ia] < values[ib] or (values[ia] == values[ib] and ia < ib)

    old = ia if older_a else ib
    young = ib if older_a else ia
    if values[young] != level:
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

    # Union by size is independent of the elder-rule representative.
    if size[ra] < size[rb]:
        parent[ra] = rb
        size[rb] += size[ra]
        elder[rb] = old
    else:
        parent[rb] = ra
        size[ra] += size[rb]
        elder[ra] = old
    return count


@_jit
def _sweep(
    values: np.ndarray, order: np.ndarray, depth: int, height: int,
    width: int, offsets: np.ndarray, h2: bool, tree: bool,
):
    n = len(values)
    outside = n
    m = n + (1 if h2 else 0)
    parent = np.full(m, -1, dtype=np.int64)
    size = np.ones(m, dtype=np.int64)
    elder = np.empty(m, dtype=np.int64)
    births = np.empty(n, dtype=values.dtype)
    deaths = np.empty(n, dtype=values.dtype)
    children = np.empty(n if tree else 0, dtype=np.int64)
    ancestors = np.empty(n if tree else 0, dtype=np.int64)
    if h2:
        parent[outside] = outside
        elder[outside] = outside
    plane = height * width
    count = 0
    for p in order:
        p = int(p)
        val = values[p]
        z = p // plane
        y = (p % plane) // width
        x = p % width
        parent[p] = p
        elder[p] = p
        if h2 and (z == 0 or z == depth - 1 or y == 0 or y == height - 1
                   or x == 0 or x == width - 1):
            count = _union(p, outside, val, values, parent, size, elder, h2,
                           outside, births, deaths, children, ancestors, count, tree)
        for k in range(len(offsets)):
            dz = int(offsets[k, 0])
            dy = int(offsets[k, 1])
            dx = int(offsets[k, 2])
            nz = z + dz
            ny = y + dy
            nx = x + dx
            if nz < 0 or nz >= depth or ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            q = nz * plane + ny * width + nx
            if parent[q] != -1:
                count = _union(p, q, val, values, parent, size, elder, h2,
                               outside, births, deaths, children, ancestors, count, tree)
    if not h2:
        root = _find(parent, 0)
        births[count] = values[elder[root]]
        deaths[count] = 0   # sentinel; not included in the finite interval table
        if tree:
            children[count] = elder[root]
            ancestors[count] = -1
        count += 1
    return (births[:count].copy(), deaths[:count].copy(),
            children[:count].copy(), ancestors[:count].copy())



def _sweep_python(
    values: np.ndarray, order: np.ndarray, depth: int, height: int,
    width: int, offsets: np.ndarray, h2: bool, tree: bool,
):
    """Fully interpreted reference of the same elder-rule sweep."""
    n = len(values)
    outside = n
    parent = [-1] * (n + int(h2))
    size = [1] * len(parent)
    elder = list(range(len(parent)))
    births = []
    deaths = []
    children = []
    ancestors = []
    if h2:
        parent[outside] = outside

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a: int, b: int, level) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        ia, ib = elder[ra], elder[rb]
        if h2 and (ia == outside or ib == outside):
            older_a = ia == outside
        elif h2:
            older_a = values[ia] > values[ib] or (values[ia] == values[ib] and ia < ib)
        else:
            older_a = values[ia] < values[ib] or (values[ia] == values[ib] and ia < ib)
        old, young = (ia, ib) if older_a else (ib, ia)
        if values[young] != level:
            if h2:
                births.append(level)
                deaths.append(values[young])
            else:
                births.append(values[young])
                deaths.append(level)
            if tree:
                children.append(young)
                ancestors.append(old)
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        elder[ra] = old

    plane = height * width
    for pp in order:
        p = int(pp)
        z, rem = divmod(p, plane)
        y, x = divmod(rem, width)
        level = values[p]
        parent[p] = p
        if h2 and (z == 0 or z == depth - 1 or y == 0 or y == height - 1 or x == 0 or x == width - 1):
            join(p, outside, level)
        for dz, dy, dx in offsets:
            nz = z + int(dz)
            ny = y + int(dy)
            nx = x + int(dx)
            if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                q = nz * plane + ny * width + nx
                if parent[q] != -1:
                    join(p, q, level)
    if not h2:
        root = find(0)
        births.append(values[elder[root]])
        deaths.append(values.dtype.type(0))
        if tree:
            children.append(elder[root])
            ancestors.append(-1)
    return (np.asarray(births, dtype=values.dtype),
            np.asarray(deaths, dtype=values.dtype),
            np.asarray(children, dtype=np.int64),
            np.asarray(ancestors, dtype=np.int64))

def _offsets(connectivity: int) -> np.ndarray:
    if connectivity not in (6, 26):
        raise ValueError("foreground_connectivity must be 6 or 26")
    return np.array([
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
        and (connectivity == 26 or abs(dz) + abs(dy) + abs(dx) == 1)
    ], dtype=np.int8)


def validate_volume(volume: np.ndarray) -> np.ndarray:
    """Require nonempty D×H×W U8/U16/F32/F64 data with finite values."""
    a = np.asarray(volume)
    if a.ndim != 3 or not all(a.shape):
        raise ValueError("volume must have nonzero (depth, height, width) dimensions")
    allowed = (np.dtype('uint8'), np.dtype('uint16'),
               np.dtype('float32'), np.dtype('float64'))
    if a.dtype not in allowed:
        raise TypeError(f"dtype {a.dtype} unsupported; use uint8, uint16, float32, or float64")
    if a.dtype.kind == 'f' and not np.isfinite(a).all():
        raise ValueError("all floating-point values must be finite (no NaN/inf)")
    # A zero-copy view is used when possible; otherwise one contiguous copy.
    return np.ascontiguousarray(a)


@dataclass(frozen=True)
class Intervals:
    """Positive intervals with one optional essential H0 interval at the end.

    births / deaths retain the native scalar dtype. death[-1] is a *dummy*
    when essential=True, and should be displayed as 'inf'.
    """

    births: np.ndarray
    deaths: np.ndarray
    dimension: Literal[0, 2]
    essential: bool = False
    branch_ids: np.ndarray | None = None
    parent_branch_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.births.shape != self.deaths.shape:
            raise ValueError("birth and death arrays have unequal lengths")
        if self.essential and (self.dimension != 0 or len(self.births) == 0):
            raise ValueError("only nonempty H0 can have an essential interval")

    def __len__(self) -> int:
        return len(self.births)

    @property
    def finite_count(self) -> int:
        return len(self) - int(self.essential)

    def pairs(self) -> list[tuple[int | float, int | float | None]]:
        """For inspection on small datasets; produces a Python object list."""
        out = []
        for i in range(len(self)):
            birth = _scalar(self.births[i])
            death = None if self.essential and i == len(self) - 1 else _scalar(self.deaths[i])
            out.append((birth, death))
        return out


def _scalar(value):
    native = value.item() if isinstance(value, np.generic) else value
    return 0.0 if isinstance(native, float) and native == 0.0 else native


def persistence(
    volume: np.ndarray, dimension: Literal[0, 2],
    foreground_connectivity: Literal[6, 26] = 6,
    *, backend: Literal['auto', 'numba', 'python', 'numba-reference'] = 'auto',
    branch_tree: bool = False,
) -> Intervals:
    """Compute exact positive H0 or H2 intervals for the sublevel filtration.

    Uses the 6/26 dual adjacency, a stable voxel-index tie break, and excludes
    zero-length intervals. Complexity: O(N log N) sorting + O(N) union-find
    memory, where N is the *entire* volume voxel count.

    backend='numba' uses a 32-bit root-cached Numba sweep when safe.
    backend='numba-reference' uses the original 64-bit Numba sweep.
    backend='python' runs the interpreted reference (slower).
    Branch identifiers are flattened voxel indices; they are available when
    branch_tree=True for generating an elder-rule branch table.
    """
    a = validate_volume(volume)
    if dimension not in (0, 2):
        raise ValueError("dimension must be 0 or 2")
    if backend not in ('auto', 'python', 'numba', 'numba-reference'):
        raise ValueError("backend must be auto, python, numba, or numba-reference")
    if backend in ('numba', 'numba-reference') and njit is None:
        raise ImportError("Numba is not installed: pip install numba")
    if backend == 'auto':
        backend = 'numba' if njit is not None else 'python'
    h2 = dimension == 2
    values = a.reshape(-1)
    # Integer counting order is stable and linear-time on sizable U8/U16
    # volumes. For small and floating volumes, stable argsort is preferable.
    order = stable_voxel_order(values, h2, counting=backend == 'numba')
    background_connectivity = 26 if foreground_connectivity == 6 else 6
    offsets = _offsets(background_connectivity if h2 else foreground_connectivity)
    if backend == 'python':
        kernel = _sweep_python
    elif backend == 'numba-reference' or values.size >= np.iinfo(np.int32).max:
        kernel = _sweep
    else:
        kernel = sweep_fast
    births, deaths, children, ancestors = kernel(
        values, order, *a.shape, offsets, h2, branch_tree
    )
    return Intervals(
        births, deaths, dimension, essential=not h2,
        branch_ids=children if branch_tree else None,
        parent_branch_ids=ancestors if branch_tree else None,
    )


def both_persistence(
    volume: np.ndarray, foreground_connectivity: Literal[6, 26] = 6,
    *, backend: Literal['auto', 'numba', 'python', 'numba-reference'] = 'auto',
    branch_tree: bool = False,
) -> tuple[Intervals, Intervals]:
    return (
        persistence(volume, 0, foreground_connectivity, backend=backend, branch_tree=branch_tree),
        persistence(volume, 2, foreground_connectivity, backend=backend, branch_tree=branch_tree),
    )


def betti_curve(intervals: Intervals) -> list[tuple[int | float, int]]:
    """Return threshold/beta changes in the same sparse form as Rust CSV.

    Each interval contributes +1 at birth and -1 at (finite) death; intervals
    have half-open semantics. Thresholds with no *change of count* are omitted.
    """
    # A dense histogram removes Python-object overhead on U8/U16 volumes.
    if intervals.births.dtype.kind == 'u' and intervals.births.dtype.itemsize in (1, 2):
        bins = 256 if intervals.births.dtype.itemsize == 1 else 65536
        deltas = np.bincount(intervals.births.astype(np.intp, copy=False), minlength=bins)
        if intervals.finite_count:
            deltas -= np.bincount(
                intervals.deaths[:intervals.finite_count].astype(np.intp, copy=False),
                minlength=bins,
            )
        change_at = np.flatnonzero(deltas)
        # A zero-sum plateau may produce a delta of zero and is skipped.
        if not len(change_at):
            return []
        values_at = np.cumsum(deltas[change_at])
        return [(int(t), int(beta)) for t, beta in zip(change_at, values_at)]
    counter: dict[int | float, int] = {}
    for birth in intervals.births:
        b = _scalar(birth)
        counter[b] = counter.get(b, 0) + 1
    for death in intervals.deaths[:intervals.finite_count]:
        d = _scalar(death)
        counter[d] = counter.get(d, 0) - 1
    beta = 0
    result = []
    previous = None
    for threshold, delta in sorted(counter.items()):
        beta += delta
        if previous != beta:
            result.append((threshold, beta))
            previous = beta
    return result


def betti_at(curve: list[tuple[int | float, int]], thresholds: np.ndarray) -> np.ndarray:
    """Evaluate a sparse Betti curve at arbitrary thresholds (including below min)."""
    t = np.asarray(thresholds)
    if not curve:
        return np.zeros_like(t, dtype=np.int64)
    values = np.array([item[0] for item in curve])
    counts = np.array([item[1] for item in curve], dtype=np.int64)
    indices = np.searchsorted(values, t, side='right') - 1
    return np.where(indices >= 0, counts[np.maximum(indices, 0)], 0)
