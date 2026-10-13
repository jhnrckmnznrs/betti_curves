"""Exact compact Z-seam certificates with optional disk-backed merge arrays.

This is an external-memory *buffering* improvement to the proven rolling-slab
algorithm, not a claim of sublinear-space exact persistence. The filtration-
weighted bipartite seam is represented by a spanning forest; every successful
union is written directly to the parent graph arrays (no separate seam edge
arrays). For U8/U16, a two-view counting sort avoids joining the two face
values into a large temporary array.

The graph's terminal vertices and elder witnesses remain explicit. They are
necessary for exact subsequent merges; disk-backed buffers bound the resident
storage of the assembled parent graph, not total peak memory.
"""
from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np

from .core import stable_voxel_order
from .hierarchical import _Graph, _jit, _find


@_jit
def _merge_sorted_nodes_to(left_ids, right_ids, left_values, right_values,
                           ids, values, lmap, rmap, shared_exterior):
    """Sorted union into caller-provided (potentially memory-mapped) arrays."""
    nl = len(left_ids) - int(shared_exterior)
    nr = len(right_ids) - int(shared_exterior)
    i = j = k = 0
    while i < nl and j < nr:
        if left_ids[i] < right_ids[j]:
            ids[k] = left_ids[i]
            values[k] = left_values[i]
            lmap[i] = k
            i += 1
        else:
            ids[k] = right_ids[j]
            values[k] = right_values[j]
            rmap[j] = k
            j += 1
        k += 1
    while i < nl:
        ids[k] = left_ids[i]
        values[k] = left_values[i]
        lmap[i] = k
        i += 1
        k += 1
    while j < nr:
        ids[k] = right_ids[j]
        values[k] = right_values[j]
        rmap[j] = k
        j += 1
        k += 1
    if shared_exterior:
        if left_ids[-1] != right_ids[-1] or left_values[-1] != right_values[-1]:
            raise AssertionError('exterior node mismatch')
        ids[k] = left_ids[-1]
        values[k] = left_values[-1]
        lmap[nl] = k
        rmap[nr] = k


@_jit
def _two_face_integer_order(values, left_base, right_base, plane, descending, bins):
    """Stable counting order from two disjoint views; no copied face values.

    Ties retain the virtual concatenation's index order, matching the older
    stable_voxel_order(concatenate((left_face, right_face)), ...).
    """
    counts = np.zeros(bins, np.int64)
    for i in range(plane):
        counts[int(values[left_base + i])] += 1
        counts[int(values[right_base + i])] += 1
    starts = np.empty(bins, np.int64)
    running = 0
    for k in range(bins):
        level = bins - 1 - k if descending else k
        starts[level] = running
        running += counts[level]
    order = np.empty(2 * plane, np.int32)
    for i in range(plane):
        key = int(values[left_base + i])
        pos = starts[key]
        order[pos] = i
        starts[key] += 1
    for i in range(plane):
        key = int(values[right_base + i])
        pos = starts[key]
        order[pos] = plane + i
        starts[key] += 1
    return order


@_jit
def _forest_into(values, order, plane, width, connectivity, left_base,
                 right_base, left_map, right_map, u, v, levels, start):
    """Build seam Kruskal forest directly into final arrays, returning length."""
    parent = np.arange(2 * plane, dtype=np.int32)
    sizes = np.ones(2 * plane, dtype=np.int32)
    active = np.zeros(2 * plane, dtype=np.bool_)
    radius = 0 if connectivity == 6 else 1
    height = plane // width
    k = start
    for pos0 in order:
        pos = int(pos0)
        active[pos] = True
        p = pos % plane
        y, x = divmod(p, width)
        root = pos
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if xx < 0 or xx >= width:
                    continue
                other = yy * width + xx + (plane if pos < plane else 0)
                if not active[other]:
                    continue
                ra, rb = _find(parent, root), _find(parent, other)
                if ra == rb:
                    continue
                if pos < plane:
                    u[k] = left_map + p
                    v[k] = right_map + yy * width + xx
                    levels[k] = values[left_base + p]
                else:
                    u[k] = left_map + yy * width + xx
                    v[k] = right_map + p
                    levels[k] = values[right_base + p]
                k += 1
                if sizes[ra] < sizes[rb]:
                    parent[ra] = rb
                    sizes[rb] += sizes[ra]
                    root = rb
                else:
                    parent[rb] = ra
                    sizes[ra] += sizes[rb]
                    root = ra
    return k


# Numba does not handle dataclass graphs as arguments. Pass the backing arrays.
@_jit
def _remap_edges(lu, lv, lw, ru, rv, rw, lmap, rmap, u, v, weights):
    n = len(lu)
    for i in range(n):
        u[i] = lmap[lu[i]]
        v[i] = lmap[lv[i]]
        weights[i] = lw[i]
    for i in range(len(ru)):
        k = n + i
        u[k] = rmap[ru[i]]
        v[k] = rmap[rv[i]]
        weights[k] = rw[i]
    return n + len(ru)


class CertificateWorkspace:
    """Owns temporary backing files for a single rolling-slab computation.

    Temporary files must remain alive until the parent reduction has completed.
    Allocating a memmap is only a spill policy; it is not an RSS hard cap.
    """

    def __init__(self, path: Path, threshold_mib: float):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold_mib) * (1024**2)
        self._count = count()
        self.peak_backed_bytes = 0
        self.mapped_allocations = 0
        self._live_paths = []
        self.peak_live_disk_bytes = 0
        self._live_disk_bytes = 0

    def allocate(self, name: str, count: int, dtype, spill: bool):
        if not spill:
            return np.empty(count, dtype=dtype)
        p = self.path / f'{next(self._count):07d}_{name}.bin'
        out = np.memmap(p, mode='w+', dtype=dtype, shape=(count,))
        self.mapped_allocations += 1
        self._live_paths.append(p)
        self._live_disk_bytes += count * np.dtype(dtype).itemsize
        self.peak_live_disk_bytes = max(self.peak_live_disk_bytes, self._live_disk_bytes)
        return out

    def release_merge_files(self):
        # Called only after the merged graph has been fully reduced and its
        # frontier has been copied into a new, independent certificate.
        # On POSIX, already-mapped pages remain accessible until closed.
        kept = []
        for path in self._live_paths:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                kept.append(path)  # Windows: final workspace cleanup
        self._live_paths = kept
        if not kept:
            self._live_disk_bytes = 0


def merge_z_certificate(left: _Graph, right: _Graph, mid: int, shape,
                        connectivity: int, h2: bool,
                        workspace: CertificateWorkspace) -> _Graph:
    """Build exact terminal-aware Z-seam certificate with fewer temporaries.

    Conservatively retains all future seam terminals and their elder witness
    nodes. Only edges redundant for *all* filtration thresholds are discarded.
    Never stores the full nine-neighbor seam edge list.
    """
    outside = int(np.prod(shape, dtype=np.int64))
    shared = bool(h2 and left.ids[-1] == outside and right.ids[-1] == outside)
    nn = len(left.ids) + len(right.ids) - int(shared)
    if nn >= np.iinfo(np.int32).max:
        raise MemoryError('certificate exceeds signed 32-bit local node indices')
    plane = int(shape[1]) * int(shape[2])
    if 2 * plane >= np.iinfo(np.int32).max:
        raise MemoryError('seam exceeds signed 32-bit forest indices')
    max_edges = len(left.u) + len(right.u) + 2 * plane - 1
    if max_edges >= np.iinfo(np.int32).max:
        raise MemoryError('certificate exceeds signed 32-bit local edge indices')
    bytes_out = (nn * (left.ids.dtype.itemsize + left.values.dtype.itemsize)
                 + max_edges * (8 + left.levels.dtype.itemsize))
    spill = bytes_out >= workspace.threshold
    ids = workspace.allocate('ids', nn, left.ids.dtype, spill)
    vals = workspace.allocate('vals', nn, left.values.dtype, spill)
    lmap = np.empty(len(left.ids), np.int32)
    rmap = np.empty(len(right.ids), np.int32)
    _merge_sorted_nodes_to(left.ids, right.ids, left.values, right.values,
                           ids, vals, lmap, rmap, shared)
    lid = (mid - 1) * plane
    rid = mid * plane
    lb = int(np.searchsorted(ids, lid))
    rb = int(np.searchsorted(ids, rid))
    if (lb + plane > nn or rb + plane > nn or ids[lb] != lid or
            ids[lb+plane-1] != lid+plane-1 or ids[rb] != rid or
            ids[rb+plane-1] != rid+plane-1):
        raise AssertionError('lost seam terminal in compact certificate')
    if vals.dtype == np.uint8 or vals.dtype == np.uint16:
        order = _two_face_integer_order(vals, lb, rb, plane, h2,
                                        256 if vals.dtype == np.uint8 else 65536)
    else:
        # Floating-point fallback: only the two faces are copied, not full
        # 3-D region intensities. This still permits exact continuous values.
        face = np.empty(2*plane, vals.dtype)
        face[:plane] = vals[lb:lb+plane]
        face[plane:] = vals[rb:rb+plane]
        order = stable_voxel_order(face, h2)
        del face
    u = workspace.allocate('u', max_edges, np.int32, spill)
    v = workspace.allocate('v', max_edges, np.int32, spill)
    levels = workspace.allocate('levels', max_edges, vals.dtype, spill)
    begin = _remap_edges(left.u, left.v, left.levels,
                         right.u, right.v, right.levels,
                         lmap, rmap, u, v, levels)
    end = _forest_into(vals, order, plane, shape[2], connectivity,
                       lb, rb, lb, rb, u, v, levels, begin)
    workspace.peak_backed_bytes = max(workspace.peak_backed_bytes, bytes_out if spill else 0)
    return _Graph(ids, vals, u[:end], v[:end], levels[:end])
