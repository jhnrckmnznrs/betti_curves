"""Exact, bounded-tile-memory divide-and-conquer H0/H2 voxel persistence.

Algorithm (graph contraction): At each region, compute the 0D filtration on its
induced adjacency graph. A younger component that has not reached the region's
boundary at the instant it dies cannot be rescued or changed by external edges:
its positive interval is final and is emitted immediately. Every other merge is
retained as a weighted edge between boundary witnesses (or an internal elder
seed), so the reduced graph has the same boundary connectivity and component
elders at every filtration level. Recurse bottom-up, adding all cross-cut edges.
At the root, attach the exterior vertex for H2 and finish the filtration.

This is not a line-for-line port of the Rust implementation. It computes the
same H0/H2 *positive interval multiset*, not necessarily the same branch-tree
representatives or order of CSV rows. Peak working memory is proportional to a
tile plus compressed interfaces (and the input's backing store), not N voxel UF
arrays; emitted interval pairs are appended directly to a disk file.

Requires Numba for efficient leaf and graph sweeps. Data are read tilewise from
an array-like source (a .npy memmap is recommended for truly out-of-core use).
"""
from __future__ import annotations

import csv
import heapq
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .core import _offsets, stable_voxel_order

try:
    from numba import njit
except ImportError:
    njit = None


def _jit(fn):
    return njit(cache=True, nogil=True)(fn) if njit else fn


@_jit
def _merge_vertex_sets(left_ids, right_ids, left_values, right_values):
    """Merge sorted disjoint boundary node sets with 32-bit maps.

    Avoids the full int64 argsort permutation and its temporary arrays.
    Child regions are disjoint, so their global IDs never overlap.
    """
    nl, nr = len(left_ids), len(right_ids)
    n = nl + nr
    ids = np.empty(n, dtype=left_ids.dtype)
    values = np.empty(n, dtype=left_values.dtype)
    lmap = np.empty(nl, dtype=np.int32)
    rmap = np.empty(nr, dtype=np.int32)
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
    return ids, values, lmap, rmap


def _id_dtype(shape, compact_ids):
    """Unsigned 32-bit global coordinates when the exterior ID also fits."""
    voxels = int(np.prod(shape, dtype=np.int64))
    return np.uint32 if compact_ids and voxels <= np.iinfo(np.uint32).max else np.int64


@_jit
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@_jit
def _leaf_sweep(values, order, depth, height, width, offsets, global_origin,
                global_height, global_width, descending):
    """Tile sweep: emit locally sealed positive intervals + interface edges."""
    n = len(values)
    parent = np.full(n, -1, np.int32)
    sizes = np.ones(n, np.int32)
    elder = np.empty(n, np.int32)
    touched = np.zeros(n, np.bool_)
    witness = np.full(n, -1, np.int64)
    born = np.empty(n, values.dtype)
    died = np.empty(n, values.dtype)
    edge_a = np.empty(n, np.int64)
    edge_b = np.empty(n, np.int64)
    edge_level = np.empty(n, values.dtype)
    b_count = 0
    e_count = 0
    plane = height * width
    global_plane = global_height * global_width
    dz0, dy0, dx0 = global_origin[0], global_origin[1], global_origin[2]
    for pp in order:
        p = int(pp)
        z, remainder = divmod(p, plane)
        y, x = divmod(remainder, width)
        gid = ((z + dz0) * global_plane + (y + dy0) * global_width + x + dx0)
        boundary = (z == 0 or z == depth - 1 or y == 0 or y == height - 1
                    or x == 0 or x == width - 1)
        parent[p] = p
        elder[p] = p
        touched[p] = boundary
        if boundary:
            witness[p] = gid
        root = p
        level = values[p]
        for k in range(len(offsets)):
            nz = z + int(offsets[k, 0])
            ny = y + int(offsets[k, 1])
            nx = x + int(offsets[k, 2])
            if nz < 0 or nz >= depth or ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            q = nz * plane + ny * width + nx
            if parent[q] == -1:
                continue
            rb = _find(parent, q)
            if rb == root:
                continue
            ea, eb = elder[root], elder[rb]
            if descending:
                a_older = values[ea] > values[eb] or (values[ea] == values[eb] and ea < eb)
            else:
                a_older = values[ea] < values[eb] or (values[ea] == values[eb] and ea < eb)
            old_root = root if a_older else rb
            young_root = rb if a_older else root
            young_idx = elder[young_root]
            old_idx = elder[old_root]
            old_touched = touched[old_root]
            young_touched = touched[young_root]
            old_anchor = witness[old_root]
            if young_touched:
                # Older component may not have touched a boundary yet. Its
                # earliest elder becomes a retained interior seed vertex.
                if not old_touched:
                    iz, rem = divmod(old_idx, plane)
                    iy, ix = divmod(rem, width)
                    old_anchor = ((iz + dz0) * global_plane + (iy + dy0) * global_width
                                  + ix + dx0)
                edge_a[e_count] = old_anchor
                edge_b[e_count] = witness[young_root]
                edge_level[e_count] = level
                e_count += 1
            elif values[young_idx] != level:
                if descending:
                    born[b_count] = level
                    died[b_count] = values[young_idx]
                else:
                    born[b_count] = values[young_idx]
                    died[b_count] = level
                b_count += 1
            if sizes[root] < sizes[rb]:
                new_root, loser = rb, root
            else:
                new_root, loser = root, rb
            parent[loser] = new_root
            sizes[new_root] += sizes[loser]
            elder[new_root] = old_idx
            touched[new_root] = old_touched or young_touched
            witness[new_root] = old_anchor if old_touched or young_touched else -1
            root = new_root
    return (born[:b_count].copy(), died[:b_count].copy(),
            edge_a[:e_count].copy(), edge_b[:e_count].copy(),
            edge_level[:e_count].copy())


@_jit
def _graph_sweep(ids, values, order, edge_a, edge_b, edge_level,
                 terminals, descending, outside_id, final):
    """Reduce weighted summary graph or close root graph, via Kruskal."""
    n = len(ids)
    parent = np.arange(n, dtype=np.int32)
    sizes = np.ones(n, dtype=np.int32)
    elder = np.arange(n, dtype=np.int32)
    touched = terminals.copy()
    witness = np.arange(n, dtype=np.int32)
    births = np.empty(n, values.dtype)
    deaths = np.empty(n, values.dtype)
    retained_a = np.empty(n, np.int32)
    retained_b = np.empty(n, np.int32)
    retained_level = np.empty(n, values.dtype)
    b_count = 0
    e_count = 0
    for kk in order:
        k = int(kk)
        ra = _find(parent, edge_a[k])
        rb = _find(parent, edge_b[k])
        if ra == rb:
            continue
        ea, eb = elder[ra], elder[rb]
        if descending and (ids[ea] == outside_id or ids[eb] == outside_id):
            a_older = ids[ea] == outside_id
        elif descending:
            a_older = values[ea] > values[eb] or (values[ea] == values[eb] and ids[ea] < ids[eb])
        else:
            a_older = values[ea] < values[eb] or (values[ea] == values[eb] and ids[ea] < ids[eb])
        old_root = ra if a_older else rb
        young_root = rb if a_older else ra
        old_elder = elder[old_root]
        young_elder = elder[young_root]
        old_terminal = touched[old_root]
        young_terminal = touched[young_root]
        old_anchor = witness[old_root] if old_terminal else old_elder
        level = edge_level[k]
        if not final and young_terminal:
            retained_a[e_count] = old_anchor
            retained_b[e_count] = witness[young_root]
            retained_level[e_count] = level
            e_count += 1
        elif values[young_elder] != level:
            if descending:
                births[b_count] = level
                deaths[b_count] = values[young_elder]
            else:
                births[b_count] = values[young_elder]
                deaths[b_count] = level
            b_count += 1
        if sizes[ra] < sizes[rb]:
            new_root, loser = rb, ra
        else:
            new_root, loser = ra, rb
        parent[loser] = new_root
        sizes[new_root] += sizes[loser]
        elder[new_root] = old_elder
        touched[new_root] = old_terminal or young_terminal
        witness[new_root] = old_anchor if old_terminal or young_terminal else -1
    # Every full region graph is connected. At the H0 final reduction, the
    # component elder is the birth of the unique essential bar.
    essential_birth = values[elder[_find(parent, 0)]]
    return (births[:b_count].copy(), deaths[:b_count].copy(),
            retained_a[:e_count].copy(), retained_b[:e_count].copy(),
            retained_level[:e_count].copy(), essential_birth)


@dataclass
class _Graph:
    ids: np.ndarray               # global voxel IDs, sorted, dtype int64
    values: np.ndarray            # native scalar dtype
    u: np.ndarray                 # dense node indices int32
    v: np.ndarray
    levels: np.ndarray            # threshold of each retained connection

    @property
    def node_count(self):
        return self.ids.size

    @property
    def edge_count(self):
        return self.u.size


@dataclass(frozen=True)
class HierarchicalResult:
    """Pair file contains consecutive (birth,death) native-dtype values, no header.

    H0 essential interval is stored separately in metadata, never in the binary
    pairs. H2 has no essential intervals. Files may be memory-mapped without
    copying via ``np.memmap(path, dtype=dtype, shape=(pair_count, 2))``.
    """
    pairs_path: Path
    metadata_path: Path
    curve_path: Path | None
    pair_count: int
    essential_birth: int | float | None
    dimension: int
    dtype: str
    peak_summary_nodes: int
    peak_summary_edges: int
    tiles: int


def _boundary(ids, box, shape):
    """Voxel IDs on the boundary of a region (outside ID is not passed)."""
    plane = shape[1] * shape[2]
    z = ids // plane
    y = (ids % plane) // shape[2]
    x = ids % shape[2]
    return ((z == box[0][0]) | (z == box[0][1] - 1) |
            (y == box[1][0]) | (y == box[1][1] - 1) |
            (x == box[2][0]) | (x == box[2][1] - 1))


def _graph_from_leaf(tile, box, full_shape, offsets, h2, sink, id_dtype=np.int64):
    flat = tile.reshape(-1)
    order = stable_voxel_order(flat, h2)
    births, deaths, gu, gv, levels = _leaf_sweep(
        flat, order, *tile.shape, offsets,
        np.array([box[i][0] for i in range(3)], np.int64),
        full_shape[1], full_shape[2], h2,
    )
    sink.write(births, deaths)
    z0, y0, x0 = (box[i][0] for i in range(3))
    d, h, w = tile.shape
    local = np.arange(tile.size, dtype=np.int64)
    z = local // (h * w)
    y = (local // w) % h
    x = local % w
    terminal = ((z == 0) | (z == d - 1) | (y == 0) | (y == h - 1) |
                (x == 0) | (x == w - 1))
    boundary_ids = ((z[terminal] + z0) * full_shape[1] * full_shape[2] +
                    (y[terminal] + y0) * full_shape[2] + x[terminal] + x0)
    nodes = np.unique(np.concatenate((boundary_ids, gu, gv))).astype(id_dtype, copy=False)
    gz = nodes // (full_shape[1] * full_shape[2]) - z0
    gy = (nodes // full_shape[2]) % full_shape[1] - y0
    gx = nodes % full_shape[2] - x0
    values = flat[(gz * h * w + gy * w + gx).astype(np.intp)]
    return _Graph(nodes, values, np.searchsorted(nodes, gu).astype(np.int32),
                  np.searchsorted(nodes, gv).astype(np.int32), levels)


def _cross_ids(box, axis, mid, shape, connectivity, id_dtype=np.int64):
    """Generate face-crossing edges into compact contiguous buffers.

    For 26-neighbor adjacency, allocate two output arrays directly rather
    than retaining up to nine large temporary arrays before concatenation.
    """
    a, b = [i for i in range(3) if i != axis]
    aa, bb = np.meshgrid(np.arange(*box[a], dtype=np.int64),
                         np.arange(*box[b], dtype=np.int64), indexing='ij')
    aa, bb = aa.ravel(), bb.ravel()
    strides = (shape[1] * shape[2], shape[2], 1)
    left_base = (mid - 1) * strides[axis] + aa * strides[a] + bb * strides[b]
    if connectivity == 6:
        return left_base.astype(id_dtype), (left_base + strides[axis]).astype(id_dtype)
    right_base = left_base + strides[axis]
    capacity = 9 * len(aa)
    left = np.empty(capacity, dtype=id_dtype)
    right = np.empty(capacity, dtype=id_dtype)
    written = 0
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            keep = ((aa + da >= box[a][0]) & (aa + da < box[a][1]) &
                    (bb + db >= box[b][0]) & (bb + db < box[b][1]))
            n = int(np.count_nonzero(keep))
            left[written:written + n] = left_base[keep]
            right[written:written + n] = (right_base[keep] +
                                          da * strides[a] + db * strides[b])
            written += n
    return left[:written], right[:written]


@_jit
def _assemble_edges_direct(ids, values, lmap, rmap,
                           left_u, left_v, left_levels,
                           right_u, right_v, right_levels,
                           starts, ends, axis, mid, strides,
                           connectivity, h2):
    """Build edges directly in final storage, without temporary seam coordinates.

    Graph IDs are sorted; Numba binary searches identify each face endpoint.
    This avoids materializing the 26-connectivity seam coordinate arrays,
    two ``searchsorted`` index arrays, and concatenate intermediates.
    """
    other_a = (axis + 1) % 3
    other_b = (axis + 2) % 3
    extent_a = ends[other_a] - starts[other_a]
    extent_b = ends[other_b] - starts[other_b]
    seam_count = (extent_a * extent_b if connectivity == 6 else
                  (3 * extent_a - 2) * (3 * extent_b - 2))
    nleft = len(left_u)
    nright = len(right_u)
    total = nleft + nright + seam_count
    u = np.empty(total, dtype=np.int32)
    v = np.empty(total, dtype=np.int32)
    levels = np.empty(total, dtype=values.dtype)
    for i in range(nleft):
        u[i] = lmap[left_u[i]]
        v[i] = lmap[left_v[i]]
        levels[i] = left_levels[i]
    for i in range(nright):
        u[nleft + i] = rmap[right_u[i]]
        v[nleft + i] = rmap[right_v[i]]
        levels[nleft + i] = right_levels[i]
    k = nleft + nright
    radius = 0 if connectivity == 6 else 1
    for a in range(starts[other_a], ends[other_a]):
        for b in range(starts[other_b], ends[other_b]):
            left_id = (mid - 1) * strides[axis] + a * strides[other_a] + b * strides[other_b]
            u_idx = np.searchsorted(ids, left_id)
            for da in range(-radius, radius + 1):
                if a + da < starts[other_a] or a + da >= ends[other_a]:
                    continue
                for db in range(-radius, radius + 1):
                    if b + db < starts[other_b] or b + db >= ends[other_b]:
                        continue
                    right_id = (mid * strides[axis] +
                                (a + da) * strides[other_a] +
                                (b + db) * strides[other_b])
                    v_idx = np.searchsorted(ids, right_id)
                    if u_idx >= len(ids) or v_idx >= len(ids) or ids[u_idx] != left_id or ids[v_idx] != right_id:
                        raise AssertionError('lost seam terminal during boundary contraction')
                    u[k] = u_idx
                    v[k] = v_idx
                    if h2:
                        levels[k] = min(values[u_idx], values[v_idx])
                    else:
                        levels[k] = max(values[u_idx], values[v_idx])
                    k += 1
    if k != total:
        raise AssertionError('seam edge-count error')
    return u, v, levels


def _merge_graphs_lowmem(left, right, box, axis, mid, shape, connectivity, h2):
    """Fused, allocation-conscious parent graph assembly."""
    ids, values, lmap, rmap = _merge_vertex_sets(
        left.ids, right.ids, left.values, right.values)
    if len(ids) >= np.iinfo(np.int32).max:
        raise MemoryError('merged boundary graph exceeds signed 32-bit local indices')
    starts = np.array([p[0] for p in box], np.int64)
    ends = np.array([p[1] for p in box], np.int64)
    strides = np.array((shape[1] * shape[2], shape[2], 1), np.int64)
    u, v, levels = _assemble_edges_direct(
        ids, values, lmap, rmap, left.u, left.v, left.levels,
        right.u, right.v, right.levels, starts, ends, axis, mid,
        strides, connectivity, h2)
    return _Graph(ids, values, u, v, levels)


def _merge_graphs(left, right, box, axis, mid, shape, connectivity, h2):
    # Each child's global IDs are unique and sorted. Fuse their merge with
    # remapping in one compiled pass rather than allocating an argsort index.
    ids, values, lmap, rmap = _merge_vertex_sets(
        left.ids, right.ids, left.values, right.values)
    if len(ids) >= np.iinfo(np.int32).max:
        raise MemoryError('merged boundary graph exceeds signed 32-bit local indices')
    seam_a, seam_b = _cross_ids(box, axis, mid, shape, connectivity, ids.dtype)
    u_seam = np.searchsorted(ids, seam_a)
    v_seam = np.searchsorted(ids, seam_b)
    # Required seam vertices were terminals in each child region.
    if (not np.array_equal(ids[u_seam], seam_a) or
            not np.array_equal(ids[v_seam], seam_b)):
        raise AssertionError('lost terminal during hierarchical compression')
    u = np.concatenate((lmap[left.u], rmap[right.u], u_seam.astype(np.int32)))
    v = np.concatenate((lmap[left.v], rmap[right.v], v_seam.astype(np.int32)))
    seam_level = (np.minimum(values[u_seam], values[v_seam]) if h2
                  else np.maximum(values[u_seam], values[v_seam]))
    weights = np.concatenate((left.levels, right.levels, seam_level))
    return _Graph(ids, values, u, v, weights)


def _reduce(graph, box, shape, h2, sink, final):
    """Prune components that cannot interact with a surrounding region."""
    outside = int(np.prod(shape))
    if final and h2:
        boundary = np.flatnonzero(_boundary(graph.ids, box, shape)).astype(np.int32)
        ix = len(graph.ids)
        ids = np.concatenate((graph.ids, np.array([outside], dtype=graph.ids.dtype)))
        values = np.concatenate((graph.values, np.array([0], dtype=graph.values.dtype)))
        u = np.concatenate((graph.u, boundary))
        v = np.concatenate((graph.v, np.full(boundary.size, ix, np.int32)))
        levels = np.concatenate((graph.levels, graph.values[boundary]))
        terminals = np.zeros(len(ids), dtype=np.bool_)
    else:
        ids, values = graph.ids, graph.values
        u, v, levels = graph.u, graph.v, graph.levels
        terminals = (np.zeros(len(ids), dtype=np.bool_) if final
                     else _boundary(ids, box, shape))
    if len(ids) >= np.iinfo(np.int32).max:
        raise MemoryError('compressed interface too large for int32; use smaller chunks / alternate backend')
    order = stable_voxel_order(levels, h2)
    births, deaths, ea, eb, ew, essential = _graph_sweep(
        ids, values, order, u, v, levels, terminals, h2, outside, final)
    sink.write(births, deaths)
    if final:
        return None, (essential.item() if hasattr(essential, "item") else essential) if not h2 else None
    keep = terminals.copy()
    keep[ea] = True
    keep[eb] = True
    selected = np.flatnonzero(keep)
    new_mapping = np.full(len(ids), -1, np.int32)
    new_mapping[selected] = np.arange(len(selected), dtype=np.int32)
    return _Graph(ids[selected], values[selected], new_mapping[ea], new_mapping[eb], ew), None


class _Sink:
    def __init__(self, out: Path, dtype, curve=True):
        self.out = out
        self.dtype = np.dtype(dtype)
        self.handle = out.open('wb')
        self.count = 0
        self.deltas = np.zeros(256 if self.dtype.itemsize == 1 else 65536, np.int64) if (
            curve and self.dtype.kind == 'u') else None

    def write(self, births, deaths):
        if not len(births):
            return
        if self.deltas is not None:
            self.deltas += np.bincount(births.astype(np.intp), minlength=len(self.deltas))
            self.deltas -= np.bincount(deaths.astype(np.intp), minlength=len(self.deltas))
        pairs = np.empty((len(births), 2), dtype=self.dtype)
        pairs[:, 0] = births
        pairs[:, 1] = deaths
        pairs.tofile(self.handle)
        self.count += len(births)

    def close(self):
        self.handle.close()


    def append_worker(self, path: Path, count: int, deltas_path: Path | None):
        """Join a worker's raw interval file without materializing its pairs."""
        if self.deltas is not None:
            if deltas_path is None:
                raise AssertionError('missing integer curve events from worker')
            self.deltas += np.load(deltas_path, allow_pickle=False)
        with path.open('rb') as incoming:
            shutil.copyfileobj(incoming, self.handle, length=4 * 1024 * 1024)
        self.count += count


def _compute_region(volume, box, shape, offsets, connectivity, h2,
                    tile_edge, sink, id_dtype, merge_strategy="fused"):
    """Compute a subtree; its exported boundary graph is its sole interface."""
    peak_nodes = peak_edges = tile_count = 0

    def stats(graph):
        nonlocal peak_nodes, peak_edges
        peak_nodes = max(peak_nodes, graph.node_count)
        peak_edges = max(peak_edges, graph.edge_count)

    def visit(region):
        nonlocal tile_count
        extents = [b - a for a, b in region]
        if max(extents) <= tile_edge:
            tile = np.ascontiguousarray(volume[tuple(slice(a, b) for a, b in region)])
            if tile.dtype.kind == 'f' and not np.isfinite(tile).all():
                raise ValueError('volume contains NaN/Inf')
            graph = _graph_from_leaf(tile, region, shape, offsets, h2, sink, id_dtype)
            stats(graph)
            tile_count += 1
            return graph
        axis = int(np.argmax(extents))
        middle = region[axis][0] + extents[axis] // 2
        lbox = list(region)
        rbox = list(region)
        lbox[axis] = (region[axis][0], middle)
        rbox[axis] = (middle, region[axis][1])
        left = visit(tuple(lbox))
        right = visit(tuple(rbox))
        merger = _merge_graphs_lowmem if merge_strategy == "fused" else _merge_graphs
        graph = merger(left, right, region, axis, middle, shape,
                       connectivity, h2)
        del left, right
        stats(graph)
        reduced, _ = _reduce(graph, region, shape, h2, sink, final=False)
        stats(reduced)
        return reduced

    graph = visit(box)
    return graph, peak_nodes, peak_edges, tile_count


def _partition_jobs(box, jobs, counter):
    """Split a binary reduction tree into a bounded number of independent jobs."""
    if jobs == 1:
        job_id = counter[0]
        counter[0] += 1
        return ('job', job_id, box)
    extents = [b - a for a, b in box]
    axis = int(np.argmax(extents))
    if extents[axis] < 2:
        # No more nonempty partitions possible.
        return _partition_jobs(box, 1, counter)
    middle = box[axis][0] + extents[axis] // 2
    left = list(box)
    right = list(box)
    left[axis] = (box[axis][0], middle)
    right[axis] = (middle, box[axis][1])
    lcount = jobs // 2
    rcount = jobs - lcount
    return ('split', box, axis, middle,
            _partition_jobs(tuple(left), lcount, counter),
            _partition_jobs(tuple(right), rcount, counter))


def _job_list(plan):
    if plan[0] == 'job':
        return [(plan[1], plan[2])]
    return _job_list(plan[4]) + _job_list(plan[5])


def _store_graph(graph, path):
    path.mkdir(parents=True, exist_ok=True)
    for field in ('ids', 'values', 'u', 'v', 'levels'):
        np.save(path / f'{field}.npy', getattr(graph, field), allow_pickle=False)


def _load_graph(path):
    # Every returned subtree summary is disk-backed, not pickled to the parent.
    arrays = [np.load(path / f'{key}.npy', mmap_mode='r', allow_pickle=False)
              for key in ('ids', 'values', 'u', 'v', 'levels')]
    return _Graph(*arrays)


def _run_subtree_array(source, box, job_directory, job_id, shape,
                       connectivity, h2, tile_edge, compact_ids, write_curve, merge_strategy="fused"):
    """One subtree, using a read-only shared array or a private memmap."""
    path = Path(job_directory) / f'job_{job_id:05d}'
    path.mkdir(parents=True, exist_ok=True)
    sink = _Sink(path / 'pairs.bin', source.dtype, curve=write_curve)
    try:
        graph, pn, pe, tiles = _compute_region(
            source, box, shape, _offsets(connectivity), connectivity, h2,
            tile_edge, sink, _id_dtype(shape, compact_ids), merge_strategy)
        _store_graph(graph, path)
        delta_path = None
        if sink.deltas is not None:
            delta_path = path / 'deltas.npy'
            np.save(delta_path, sink.deltas, allow_pickle=False)
        return (job_id, path, sink.count, delta_path, pn, pe, tiles)
    finally:
        sink.close()


def _run_subtree_job(source_path, box, job_directory, job_id, shape,
                     connectivity, h2, tile_edge, compact_ids, write_curve, merge_strategy="fused"):
    """Process entry: reopen source instead of pickling an image."""
    source = np.load(source_path, mmap_mode='r', allow_pickle=False)
    return _run_subtree_array(source, box, job_directory, job_id, shape,
                              connectivity, h2, tile_edge, compact_ids, write_curve, merge_strategy)


def _prepare_worker_source(volume, scratch: Path):
    """Reuse compatible .npy maps, otherwise spool source one z-slice at a time."""
    if isinstance(volume, np.memmap):
        filename = getattr(volume, 'filename', None)
        if filename is not None and Path(filename).suffix.lower() == '.npy':
            try:
                check = np.load(filename, mmap_mode='r', allow_pickle=False)
                if (check.shape == volume.shape and check.dtype == volume.dtype and
                        check.offset == volume.offset and
                        check.strides == volume.strides):
                    return Path(filename), False
            except (OSError, ValueError):
                pass
    dest = scratch / 'worker_volume.npy'
    out = np.lib.format.open_memmap(dest, mode='w+', dtype=volume.dtype, shape=volume.shape)
    try:
        for z in range(volume.shape[0]):
            out[z] = volume[z]
        out.flush()
    finally:
        del out
    return dest, True


def _compute_parallel(volume, root_box, shape, connectivity, h2, tile_edge,
                      compact_ids, workers, sink, write_curve, scratch,
                      parallel_backend, merge_strategy="fused"):
    """Compute independent boundary summaries and reconcile in tree order.

    Threads share a read-only input and run Numba kernels without the GIL;
    processes reopen a .npy input to avoid shipping an entire image by IPC.
    In either case, results travel through disk instead of pickled graphs.
    """
    plan = _partition_jobs(root_box, min(workers, int(np.prod(shape))), [0])
    jobs = _job_list(plan)
    job_directory = scratch / 'jobs'
    job_directory.mkdir()
    spooled = False
    if parallel_backend == 'processes':
        source_path, spooled = _prepare_worker_source(volume, scratch)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
    with pool:
        if parallel_backend == 'processes':
            futures = [pool.submit(_run_subtree_job, source_path, box, job_directory,
                                   job_id, shape, connectivity, h2, tile_edge,
                                   compact_ids, write_curve, merge_strategy)
                       for job_id, box in jobs]
        else:
            futures = [pool.submit(_run_subtree_array, volume, box, job_directory,
                                   job_id, shape, connectivity, h2, tile_edge,
                                   compact_ids, write_curve, merge_strategy)
                       for job_id, box in jobs]
        results = {}
        for future in futures:
            entry = future.result()
            results[entry[0]] = entry

    peak_nodes = peak_edges = tiles = 0

    def visit(tree):
        nonlocal peak_nodes, peak_edges, tiles
        if tree[0] == 'job':
            job_id = tree[1]
            _, path, count, delta_path, pn, pe, count_tiles = results[job_id]
            sink.append_worker(path / 'pairs.bin', count, delta_path)
            graph = _load_graph(path)
            peak_nodes = max(peak_nodes, pn, graph.node_count)
            peak_edges = max(peak_edges, pe, graph.edge_count)
            tiles += count_tiles
            return graph
        _, box, axis, mid, ltree, rtree = tree
        left = visit(ltree)
        right = visit(rtree)
        merger = _merge_graphs_lowmem if merge_strategy == "fused" else _merge_graphs
        merged = merger(left, right, box, axis, mid, shape, connectivity, h2)
        del left, right
        peak_nodes = max(peak_nodes, merged.node_count)
        peak_edges = max(peak_edges, merged.edge_count)
        reduced, _ = _reduce(merged, box, shape, h2, sink, final=False)
        peak_nodes = max(peak_nodes, reduced.node_count)
        peak_edges = max(peak_edges, reduced.edge_count)
        return reduced

    graph = visit(plan)
    return graph, peak_nodes, peak_edges, tiles, spooled


def _graph_storage_bytes(graph):
    return sum(getattr(graph, field).nbytes for field in ('ids', 'values', 'u', 'v', 'levels'))


def _compute_adaptive(volume, root_box, shape, connectivity, h2, tile_edge,
                      compact_ids, workers, sink, write_curve, scratch,
                      parallel_backend, merge_strategy, task_factor,
                      pending_summary_mib):
    """Demand-driven workers, immediate sibling reductions, bounded pending RAM.

    Divide the root into more tasks than workers, enqueue the largest first,
    replenish worker slots as tasks finish, and reduce sibling summaries as soon
    as both are ready. Worker-to-parent payloads are paths, not large graphs.
    Completed but unpaired parent summaries are spilled when the pending-RAM
    budget is exceeded. This is *task-level* dynamic scheduling, not preemption
    of a running tile or an optimal global task graph executor.
    """
    if task_factor < 1:
        raise ValueError('task_factor must be positive')
    if pending_summary_mib < 0:
        raise ValueError('pending_summary_mib must be nonnegative')
    # Do not split regions smaller than an ordinary tile just to satisfy an
    # oversubscription target. Bisection can produce somewhat more leaf tiles
    # than the simple geometric lower bound below.
    minimum_tiles = 1
    for d in shape:
        minimum_tiles *= (d + tile_edge - 1) // tile_edge
    target_jobs = min(workers * task_factor, minimum_tiles)
    plan = _partition_jobs(root_box, target_jobs, [0])
    jobs = _job_list(plan)
    jobs.sort(key=lambda item: -int(np.prod([b - a for a, b in item[1]])))
    job_directory = scratch / 'jobs'
    job_directory.mkdir()
    max_pending = int(pending_summary_mib * 1024 * 1024)
    pending = {}
    ram_bytes = 0
    peak_pending = 0
    parent_of = {}
    leaf_of = {}

    def describe(tree, parent=None):
        identity = id(tree)
        if parent is not None:
            parent_of[identity] = parent
        if tree[0] == 'job':
            leaf_of[tree[1]] = tree
        else:
            describe(tree[4], tree)
            describe(tree[5], tree)

    describe(plan)
    peak_nodes = peak_edges = tiles = 0
    merges = spills = 0
    sequence = [0]
    spooled = False
    if parallel_backend == 'processes':
        source_path, spooled = _prepare_worker_source(volume, scratch)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
    else:
        pool = ThreadPoolExecutor(max_workers=workers)

    def load_ref(ref):
        return _load_graph(ref) if isinstance(ref, Path) else ref

    def reduce_available(tree, ref):
        nonlocal peak_nodes, peak_edges, ram_bytes, peak_pending, merges, spills
        while True:
            identity = id(tree)
            if isinstance(ref, _Graph):
                ram_bytes += _graph_storage_bytes(ref)
            pending[identity] = ref
            parent = parent_of.get(identity)
            if parent is None:
                break
            lid = id(parent[4])
            rid = id(parent[5])
            if lid not in pending or rid not in pending:
                break
            lref, rref = pending.pop(lid), pending.pop(rid)
            ram_bytes -= sum(_graph_storage_bytes(item) for item in (lref, rref)
                             if isinstance(item, _Graph))
            left, right = load_ref(lref), load_ref(rref)
            merger = _merge_graphs_lowmem if merge_strategy == 'fused' else _merge_graphs
            _, box, axis, mid, _, _ = parent
            joined = merger(left, right, box, axis, mid, shape, connectivity, h2)
            peak_nodes = max(peak_nodes, joined.node_count)
            peak_edges = max(peak_edges, joined.edge_count)
            del left, right
            ref, _ = _reduce(joined, box, shape, h2, sink, final=False)
            del joined
            peak_nodes = max(peak_nodes, ref.node_count)
            peak_edges = max(peak_edges, ref.edge_count)
            merges += 1
            tree = parent
        # Bound the bytes held by *waiting* resident arrays. Worker graphs and
        # spilled intermediate summaries are mmapped on demand at merge time.
        while ram_bytes > max_pending:
            candidates = [(k, v, _graph_storage_bytes(v)) for k, v in pending.items()
                          if isinstance(v, _Graph) and k != id(plan)]
            if not candidates:
                # The final root graph is consumed immediately after workers
                # exit; spilling it would just add an unnecessary I/O roundtrip.
                break
            # Spill the largest *waiting* graph to reduce file writes.
            key, biggest, _ = max(candidates, key=lambda item: item[2])
            path = scratch / f'pending_{sequence[0]:06d}'
            sequence[0] += 1
            _store_graph(biggest, path)
            pending[key] = path
            ram_bytes -= _graph_storage_bytes(biggest)
            spills += 1
            del biggest
        peak_pending = max(peak_pending, ram_bytes)

    next_job = iter(jobs)
    futures = {}

    def submit(pool):
        try:
            job_id, box = next(next_job)
        except StopIteration:
            return False
        if parallel_backend == 'processes':
            future = pool.submit(_run_subtree_job, source_path, box, job_directory,
                                 job_id, shape, connectivity, h2, tile_edge,
                                 compact_ids, write_curve, merge_strategy)
        else:
            future = pool.submit(_run_subtree_array, volume, box, job_directory,
                                 job_id, shape, connectivity, h2, tile_edge,
                                 compact_ids, write_curve, merge_strategy)
        futures[future] = job_id
        return True

    with pool:
        for _ in range(min(len(jobs), workers * 2)):
            submit(pool)
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                expected_id = futures.pop(future)
                job_id, path, count, delta_path, pn, pe, count_tiles = future.result()
                if job_id != expected_id:
                    raise AssertionError('worker ID mismatch')
                sink.append_worker(path / 'pairs.bin', count, delta_path)
                # These disk files are never needed again once interval events
                # have been folded into the main binary stream/histogram.
                (path / 'pairs.bin').unlink()
                if delta_path is not None:
                    delta_path.unlink()
                peak_nodes = max(peak_nodes, pn)
                peak_edges = max(peak_edges, pe)
                tiles += count_tiles
                reduce_available(leaf_of[job_id], path)
                submit(pool)
    root = pending.get(id(plan))
    if root is None or len(pending) != 1:
        raise AssertionError('incomplete adaptive hierarchy')
    return (load_ref(root), peak_nodes, peak_edges, tiles, spooled,
            len(jobs), merges, spills, peak_pending)


def _write_float_curve_external(pairs_path: Path, curve_path: Path, *, dtype,
                                pair_count: int, dimension: int,
                                essential_birth, chunk_pairs: int = 1_000_000,
                                merge_fan_in: int = 32) -> None:
    """Disk-backed event sort and k-way merge for F32/F64 sparse Betti curves.

    Sorting each chunk uses O(chunk_pairs) RAM, and the binary runs are merged
    in bounded-fan-in passes. No global dictionary of distinct float values.
    """
    event_dtype = np.dtype([('level', np.dtype(dtype)), ('delta', '<i8')])
    if pair_count:
        pairs = np.memmap(pairs_path, mode='r', dtype=dtype, shape=(pair_count, 2))
    else:
        pairs = None

    def merge_runs(run_paths, dest: Path, csv_output: bool):
        opened = [np.memmap(p, mode='r', dtype=event_dtype) for p in run_paths]
        heap = []
        for i, run in enumerate(opened):
            if len(run):
                heapq.heappush(heap, (float(run[0]['level']), i, 0, int(run[0]['delta'])))
        if csv_output:
            handle = dest.open('w', newline='', encoding='utf-8')
            writer = csv.writer(handle)
            writer.writerow(['threshold', 'betti0' if dimension == 0 else 'betti2'])
        else:
            handle = dest.open('wb')
            writer = None
        beta = 0
        pending = []
        with handle:
            while heap:
                val, i, pos, delta = heapq.heappop(heap)
                pos += 1
                if pos < len(opened[i]):
                    row = opened[i][pos]
                    heapq.heappush(heap, (float(row['level']), i, pos, int(row['delta'])))
                while heap and heap[0][0] == val:
                    _, j, ix, d = heapq.heappop(heap)
                    delta += d
                    ix += 1
                    if ix < len(opened[j]):
                        row = opened[j][ix]
                        heapq.heappush(heap, (float(row['level']), j, ix, int(row['delta'])))
                if not delta:
                    continue
                if csv_output:
                    beta += delta
                    writer.writerow((val, beta))
                else:
                    pending.append((val, delta))
                    if len(pending) >= 8192:
                        np.array(pending, dtype=event_dtype).tofile(handle)
                        pending.clear()
            if pending:
                np.array(pending, dtype=event_dtype).tofile(handle)
        # Ensure all memmap handles can be collected before removing run files.
        del opened

    with tempfile.TemporaryDirectory(prefix='betti_curve_runs_', dir=curve_path.parent) as tmp:
        folder = Path(tmp)
        paths = []
        for start in range(0, pair_count, chunk_pairs):
            part = np.asarray(pairs[start:min(pair_count, start + chunk_pairs)])
            level = np.concatenate((part[:, 0], part[:, 1]))
            delta = np.concatenate((np.ones(len(part), np.int64),
                                    -np.ones(len(part), np.int64)))
            order = np.argsort(level, kind='stable')
            level = level[order]
            delta = delta[order]
            boundaries = np.r_[0, np.flatnonzero(level[1:] != level[:-1]) + 1]
            aggregated = np.add.reduceat(delta, boundaries)
            mask = aggregated != 0
            events = np.empty(np.count_nonzero(mask), dtype=event_dtype)
            events['level'] = level[boundaries[mask]]
            events['delta'] = aggregated[mask]
            dest = folder / f'run_{len(paths):06d}.bin'
            events.tofile(dest)
            if len(events):
                paths.append(dest)
        if dimension == 0:
            event = np.array([(essential_birth, 1)], dtype=event_dtype)
            dest = folder / 'essential.bin'
            event.tofile(dest)
            paths.append(dest)
        if not paths:
            with curve_path.open('w', newline='', encoding='utf-8') as handle:
                csv.writer(handle).writerow(['threshold', 'betti2'])
            return
        pass_number = 0
        while len(paths) > merge_fan_in:
            next_paths = []
            for start in range(0, len(paths), merge_fan_in):
                subset = paths[start:start + merge_fan_in]
                dest = folder / f'merge_{pass_number}_{start}.bin'
                merge_runs(subset, dest, False)
                next_paths.append(dest)
            for old in paths:
                old.unlink()
            paths = next_paths
            pass_number += 1
        merge_runs(paths, curve_path, True)


def _validate_source(volume):
    if len(volume.shape) != 3 or min(volume.shape) <= 0:
        raise ValueError('expected nonempty 3D volume')
    if np.dtype(volume.dtype) not in tuple(map(np.dtype, ('uint8','uint16','float32','float64'))):
        raise TypeError('requires U8, U16, F32 or F64 data')
    if int(np.prod(volume.shape)) >= np.iinfo(np.int64).max:
        raise ValueError('voxel count overflows 64-bit global IDs')


def hierarchical_persistence(
    volume, dimension: Literal[0, 2], output_dir: str | Path, *,
    foreground_connectivity: Literal[6, 26] = 6,
    tile_edge: int = 48,
    write_curve: bool = True,
    parallel_workers: int = 1,
    compact_ids: bool = True,
    parallel_backend: Literal['threads', 'processes'] = 'threads',
    scheduling: Literal['adaptive', 'fixed'] = 'fixed',
    task_factor: int = 3,
    pending_summary_mib: float = 128,
    merge_strategy: Literal['fused', 'legacy'] = 'fused',
) -> HierarchicalResult:
    """Exact disk-backed H0/H2 intervals on a large in-memory/memory-mapped 3D array.

    No whole-volume sort or union-find buffers. Interval pairs append to a raw
    native-endian two-column binary file, and integer sparse curves to CSV.
    Memory scales with the largest leaf tile + compressed subregion interfaces.

    Float sparse curves are computed using a bounded-memory external merge-sort
    over interval endpoints. This can entail substantial temporary disk I/O.
    Branch trees are not exported by this hierarchical backend.

    ``parallel_workers > 1`` computes disjoint subtrees concurrently. Threads
    (default) share a read-only source, with GIL-free Numba kernels. Processes
    reopen a shared .npy file, avoiding pickling an entire image; other input
    types are first spooled slice-by-slice. In both modes, worker summaries and
    interval streams travel via temporary disk files. Additional workers can
    increase peak RAM; parallelism is not always faster on small volumes.
    ``compact_ids`` uses uint32 boundary IDs when the global outside ID fits.
    """
    if njit is None:
        raise ImportError('hierarchical backend requires numba; pip install numba')
    _validate_source(volume)
    if dimension not in (0, 2):
        raise ValueError('dimension must be 0 or 2')
    if foreground_connectivity not in (6, 26):
        raise ValueError('foreground_connectivity must be 6 or 26')
    if tile_edge < 2:
        raise ValueError('tile_edge must be >= 2')
    if not isinstance(parallel_workers, int) or parallel_workers < 1:
        raise ValueError('parallel_workers must be a positive integer')
    if parallel_backend not in ('threads', 'processes'):
        raise ValueError('parallel_backend must be threads or processes')
    if scheduling not in ('adaptive', 'fixed'):
        raise ValueError('scheduling must be adaptive or fixed')
    if merge_strategy not in ('fused', 'legacy'):
        raise ValueError('merge_strategy must be fused or legacy')
    if not isinstance(task_factor, int) or task_factor < 1:
        raise ValueError('task_factor must be positive')
    if not np.isfinite(pending_summary_mib) or pending_summary_mib < 0:
        raise ValueError('pending_summary_mib must be nonnegative')
    shape = tuple(int(x) for x in volume.shape)
    h2 = dimension == 2
    connectivity = ((26 if foreground_connectivity == 6 else 6) if h2
                    else foreground_connectivity)
    offsets = _offsets(connectivity)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = 'h2' if h2 else 'h0'
    pairs_path = out / f'{key}_pairs.bin'
    metadata_path = out / f'{key}_metadata.json'
    curve_path = out / f'{key}_betti_curve.csv' if write_curve else None
    sink = _Sink(pairs_path, volume.dtype, curve=write_curve)
    peak_nodes, peak_edges, tiles = 0, 0, 0
    outer_box = tuple((0, d) for d in shape)
    spooled = False
    scheduled_jobs = 1
    adaptive_merges = summary_spills = peak_pending_bytes = 0

    try:
        if parallel_workers == 1:
            graph, peak_nodes, peak_edges, tiles = _compute_region(
                volume, outer_box, shape, offsets, connectivity, h2,
                tile_edge, sink, _id_dtype(shape, compact_ids), merge_strategy)
        else:
            # Intermediate job artifacts disappear once the parent has
            # consumed them. Files remain available throughout reduction.
            with tempfile.TemporaryDirectory(prefix='betti_workers_', dir=out) as workdir:
                if scheduling == 'adaptive':
                    (graph, peak_nodes, peak_edges, tiles, spooled, scheduled_jobs,
                     adaptive_merges, summary_spills, peak_pending_bytes) = _compute_adaptive(
                        volume, outer_box, shape, connectivity, h2, tile_edge,
                        compact_ids, parallel_workers, sink, write_curve,
                        Path(workdir), parallel_backend, merge_strategy,
                        task_factor, pending_summary_mib)
                else:
                    graph, peak_nodes, peak_edges, tiles, spooled = _compute_parallel(
                        volume, outer_box, shape, connectivity, h2, tile_edge,
                        compact_ids, parallel_workers, sink, write_curve,
                        Path(workdir), parallel_backend, merge_strategy)
                    scheduled_jobs = min(parallel_workers, int(np.prod(shape)))
        
        _, essential = _reduce(graph, outer_box, shape, h2, sink, final=True)
        if sink.deltas is not None and not h2:
            sink.deltas[int(essential)] += 1
        if curve_path is not None and sink.deltas is not None:
            delta = sink.deltas
            changes = np.flatnonzero(delta)
            with curve_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['threshold', 'betti2' if h2 else 'betti0'])
                beta = 0
                for t in changes:
                    beta += int(delta[t])
                    writer.writerow([int(t), beta])
        elif curve_path is not None:
            sink.handle.flush()
            _write_float_curve_external(pairs_path, curve_path, dtype=volume.dtype,
                                        pair_count=sink.count, dimension=dimension,
                                        essential_birth=essential)
        native_essential = essential.item() if hasattr(essential, 'item') else essential
        report = dict(algorithm='hierarchical-boundary-contraction',
                      dimensions=shape, dtype=str(volume.dtype), dimension=dimension,
                      foreground_connectivity=foreground_connectivity,
                      tile_edge=tile_edge, finite_pairs=sink.count,
                      parallel_workers=parallel_workers, parallel_backend=parallel_backend,
                      scheduling=scheduling, task_factor=task_factor,
                      scheduled_jobs=scheduled_jobs, adaptive_merges=adaptive_merges,
                      summary_spills=summary_spills, peak_pending_summary_bytes=peak_pending_bytes,
                      pending_summary_mib=pending_summary_mib, merge_strategy=merge_strategy,
                      compact_ids=compact_ids,
                      global_id_dtype=np.dtype(_id_dtype(shape, compact_ids)).name,
                      parallel_input_spooled=spooled,
                      essential_birth=native_essential, binary_pair_layout='two consecutive native-endian values per pair',
                      peak_summary_nodes=peak_nodes, peak_summary_edges=peak_edges,
                      tiles=tiles, curve=str(curve_path) if curve_path is not None else None)
        with metadata_path.open('w') as handle:
            json.dump(report, handle, indent=2)
        return HierarchicalResult(pairs_path, metadata_path, curve_path,
                                  sink.count, native_essential, dimension,
                                  str(volume.dtype), peak_nodes, peak_edges, tiles)
    finally:
        sink.close()


def load_hierarchical_pairs(result: HierarchicalResult) -> np.ndarray:
    """Return a zero-copy disk view; avoid converting this to an array for huge data."""
    return np.memmap(result.pairs_path, mode='r', dtype=result.dtype,
                     shape=(result.pair_count, 2)) if result.pair_count else np.empty((0,2), dtype=result.dtype)


def export_hierarchical_pairs_csv(result: HierarchicalResult,
                                  output_path: str | Path,
                                  chunk_pairs: int = 100_000) -> Path:
    """Convert binary pairs to familiar CSV in fixed-size chunks.

    For very large barcodes CSV can be much larger and slower than the binary
    pair file. The H0 essential interval is appended as (birth, inf).
    """
    if chunk_pairs < 1:
        raise ValueError('chunk_pairs must be positive')
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pairs = load_hierarchical_pairs(result)
    with destination.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('birth', 'death'))
        for start in range(0, result.pair_count, chunk_pairs):
            end = min(start + chunk_pairs, result.pair_count)
            writer.writerows(pairs[start:end].tolist())
        if result.dimension == 0:
            writer.writerow((result.essential_birth, 'inf'))
    return destination
