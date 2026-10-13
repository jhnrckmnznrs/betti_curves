"""Exact H0/H2 sequential-Z slab persistence with minimal surviving frontier.

The source is accessed in tiles; it need not be a resident numpy ndarray. At
any intermediate step only connections crossing the *next Z plane* can matter.
For H2 the global exterior is introduced early and is carried as a permanent
terminal, so physical side faces from processed slabs may be discarded.

The contraction proof is the same as hierarchical._graph_sweep: each region is
replaced by its filtration-preserving boundary/elder witness graph. Unlike a
3-D recursive hierarchy, past X/Y/Z external faces are never retained merely
because they bound an already processed prefix.

The default rolling cross-Z seam uses a filtration-preserving spanning forest.
Optional boundary-aware recursion prunes physical X/Y faces within a slab, and
optional internal forests avoid the 9-edges-per-pixel 26-neighbor allocation.
Neither approach imposes a strict bound on peak graph memory.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from .core import _offsets, stable_voxel_order
from .certificate import CertificateWorkspace, merge_z_certificate
from .paged import merge_z_paged, reduce_frontier_paged
from .implicit_frontier import ImplicitFrontier, pack_frontier, merge_z_implicit
from .parent_stream import stream_parent_reduce
from .hierarchical import (
    HierarchicalResult, _Graph, _Sink, _compute_region, _graph_sweep,
    _id_dtype, _merge_graphs, _merge_graphs_lowmem, _merge_vertex_sets,
    _assemble_edges_direct, _graph_from_leaf,
    _validate_source, _find, _jit,
    _write_float_curve_external, njit,
)



@_jit
def _seam_spanning_forest(face_values, order, width, connectivity, descending,
                          left_offset, right_offset):
    """Kruskal MSF of cross-Z edges without allocating 9 edges per pixel.

    Edge filtration value is max(endpoint values) for H0 (ascending) and
    min(endpoint values) for H2 (descending). Consequently, processing face
    pixels by activation order makes each newly usable edge have precisely
    the current pixel's value. Union all active opposite-face neighbors;
    retaining only successful joins yields an exact filtration-preserving
    minimum/maximum spanning *forest* of the bipartite seam graph.
    """
    plane = len(face_values) // 2
    parent = np.arange(2 * plane, dtype=np.int32)
    sizes = np.ones(2 * plane, dtype=np.int32)
    active = np.zeros(2 * plane, dtype=np.bool_)
    # Every successful merge reduces the number of components by one.
    out_u = np.empty(2 * plane - 1, np.int32)
    out_v = np.empty(2 * plane - 1, np.int32)
    out_level = np.empty(2 * plane - 1, face_values.dtype)
    radius = 0 if connectivity == 6 else 1
    height = plane // width
    count = 0
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
                neighbor = (yy * width + xx) + (plane if pos < plane else 0)
                if not active[neighbor]:
                    continue
                rb = _find(parent, neighbor)
                ra = _find(parent, root)
                if ra == rb:
                    continue
                if pos < plane:
                    out_u[count] = left_offset + p
                    out_v[count] = right_offset + (yy * width + xx)
                else:
                    out_u[count] = left_offset + (yy * width + xx)
                    out_v[count] = right_offset + p
                out_level[count] = face_values[pos]
                count += 1
                if sizes[ra] < sizes[rb]:
                    parent[ra] = rb
                    sizes[rb] += sizes[ra]
                    root = rb
                else:
                    parent[rb] = ra
                    sizes[ra] += sizes[rb]
                    root = ra
    return out_u[:count].copy(), out_v[:count].copy(), out_level[:count].copy()


def _merge_z_forest(left: _Graph, right: _Graph, mid: int,
                    shape, connectivity: int, h2: bool) -> _Graph:
    """Stitch successive full XY slabs using a minimal cross-seam forest."""
    outside = int(np.prod(shape, dtype=np.int64))
    shared_exterior = (h2 and left.ids[-1] == outside and right.ids[-1] == outside)
    if shared_exterior:
        ids, values, lmap, rmap = _merge_vertex_sets(
            left.ids[:-1], right.ids[:-1], left.values[:-1], right.values[:-1])
        if left.values[-1] != right.values[-1]:
            raise AssertionError('inconsistent exterior values')
        outside_pos = len(ids)
        ids = np.concatenate((ids, left.ids[-1:]))
        values = np.concatenate((values, left.values[-1:]))
        lmap = np.concatenate((lmap, np.array([outside_pos], np.int32)))
        rmap = np.concatenate((rmap, np.array([outside_pos], np.int32)))
    else:
        ids, values, lmap, rmap = _merge_vertex_sets(
            left.ids, right.ids, left.values, right.values)
    if len(ids) >= np.iinfo(np.int32).max:
        raise MemoryError('combined graph exceeds int32 local node limit')
    plane = int(shape[1]) * int(shape[2])
    if 2 * plane >= np.iinfo(np.int32).max:
        raise MemoryError('cross-slab plane exceeds signed int32 local forest indices')
    left_start_id = (mid - 1) * plane
    right_start_id = mid * plane
    lbase = int(np.searchsorted(ids, left_start_id))
    rbase = int(np.searchsorted(ids, right_start_id))
    # Both faces must be complete terminal planes. If one was discarded, the
    # reduction is invalid; do not silently proceed with wrong indices.
    if (rbase + plane > len(ids) or lbase + plane > len(ids) or
            ids[lbase] != left_start_id or ids[lbase + plane - 1] != left_start_id + plane - 1 or
            ids[rbase] != right_start_id or ids[rbase + plane - 1] != right_start_id + plane - 1):
        raise AssertionError('lost full seam face during rolling contraction')
    face_values = np.concatenate((values[lbase:lbase+plane],
                                  values[rbase:rbase+plane]))
    order = stable_voxel_order(face_values, h2)
    su, sv, sw = _seam_spanning_forest(face_values, order, shape[2],
                                      connectivity, h2, lbase, rbase)
    del face_values, order
    nleft, nright, nseam = len(left.u), len(right.u), len(su)
    # Work with Numba and dense arrays; avoid building 9*XY face-edge arrays.
    u = np.empty(nleft+nright+nseam, np.int32)
    v = np.empty(nleft+nright+nseam, np.int32)
    w = np.empty(nleft+nright+nseam, values.dtype)
    u[:nleft], v[:nleft], w[:nleft] = lmap[left.u], lmap[left.v], left.levels
    u[nleft:nleft+nright], v[nleft:nleft+nright], w[nleft:nleft+nright] = (
        rmap[right.u], rmap[right.v], right.levels)
    u[nleft+nright:], v[nleft+nright:], w[nleft+nright:] = su, sv, sw
    return _Graph(ids, values, u, v, w)

def _combine_shared_nodes(left: _Graph, right: _Graph, outside: int):
    """Combine disjoint regional witnesses; coalesce the shared exterior ID.

    Both child index maps must refer to exactly the same exterior node when
    present on both sides. Other global voxel IDs are always disjoint.
    """
    shared = left.ids[-1] == outside and right.ids[-1] == outside
    if shared:
        ids, values, lmap, rmap = _merge_vertex_sets(
            left.ids[:-1], right.ids[:-1], left.values[:-1], right.values[:-1])
        if left.values[-1] != right.values[-1]:
            raise AssertionError('inconsistent outside sentinel values')
        end = len(ids)
        ids = np.concatenate((ids, left.ids[-1:]))
        values = np.concatenate((values, left.values[-1:]))
        lmap = np.concatenate((lmap, np.array([end], np.int32)))
        rmap = np.concatenate((rmap, np.array([end], np.int32)))
        return ids, values, lmap, rmap
    return _merge_vertex_sets(left.ids, right.ids, left.values, right.values)


def _merge_internal_forest(left: _Graph, right: _Graph, box,
                           axis: int, mid: int, shape, connectivity: int,
                           h2: bool) -> _Graph:
    """Forest-compress 26-connected interior cross-cut edges on the fly.

    Internal faces are not contiguous in global voxel-ID space for X/Y cuts.
    Instead build their two row-major index maps into the sorted union of
    child graph nodes, run the general bipartite seam forest in local face
    coordinates, and remap ONLY the successful forest edges to graph IDs.
    """
    ids, values, lmap, rmap = _combine_shared_nodes(
        left, right, int(np.prod(shape, dtype=np.int64)))
    if len(ids) >= np.iinfo(np.int32).max:
        raise MemoryError('internal forest graph exceeds int32 node capacity')
    other_a, other_b = [k for k in range(3) if k != axis]
    extent_a = box[other_a][1] - box[other_a][0]
    extent_b = box[other_b][1] - box[other_b][0]
    stride = (int(shape[1])*int(shape[2]), int(shape[2]), 1)
    va = np.arange(*box[other_a], dtype=np.int64)
    vb = np.arange(*box[other_b], dtype=np.int64)
    left_ids = ((mid-1) * stride[axis] +
                (va[:,None] * stride[other_a] + vb[None,:] * stride[other_b])).ravel()
    right_ids = left_ids + stride[axis]
    left_index = np.searchsorted(ids, left_ids).astype(np.int32)
    right_index = np.searchsorted(ids, right_ids).astype(np.int32)
    if (np.any(left_index >= len(ids)) or np.any(right_index >= len(ids)) or
            not np.array_equal(ids[left_index], left_ids) or
            not np.array_equal(ids[right_index], right_ids)):
        raise AssertionError('missing internal seam terminal')
    face_map = np.concatenate((left_index, right_index))
    face_values = values[face_map]
    order = stable_voxel_order(face_values, h2)
    ulocal, vlocal, weights = _seam_spanning_forest(
        face_values, order, extent_b, connectivity, h2, 0, extent_a*extent_b)
    left_edges, right_edges, ns = len(left.u), len(right.u), len(ulocal)
    u = np.empty(left_edges + right_edges + ns, np.int32)
    v = np.empty_like(u)
    levels = np.empty(len(u), dtype=values.dtype)
    u[:left_edges], v[:left_edges], levels[:left_edges] = (
        lmap[left.u], lmap[left.v], left.levels)
    u[left_edges:left_edges+right_edges], v[left_edges:left_edges+right_edges], levels[left_edges:left_edges+right_edges] = (
        rmap[right.u], rmap[right.v], right.levels)
    u[left_edges+right_edges:] = face_map[ulocal]
    v[left_edges+right_edges:] = face_map[vlocal]
    levels[left_edges+right_edges:] = weights
    return _Graph(ids, values, u, v, levels)


def _merge_regions_shared(left: _Graph, right: _Graph, box, axis: int, mid: int,
                          shape, connectivity: int, h2: bool) -> _Graph:
    ids, values, lmap, rmap = _combine_shared_nodes(
        left, right, int(np.prod(shape, dtype=np.int64)))
    if len(ids) >= np.iinfo(np.int32).max:
        raise MemoryError('regional graph exceeds int32 local node limit')
    starts = np.array([p[0] for p in box], np.int64)
    ends = np.array([p[1] for p in box], np.int64)
    strides = np.array([shape[1] * shape[2], shape[2], 1], np.int64)
    u, v, levels = _assemble_edges_direct(
        ids, values, lmap, rmap, left.u, left.v, left.levels,
        right.u, right.v, right.levels, starts, ends, axis, mid,
        strides, connectivity, h2)
    return _Graph(ids, values, u, v, levels)


def _reduce_physical_sides(graph: _Graph, box, shape, h2: bool,
                           sink: _Sink) -> _Graph:
    """Only preserve sides that can see UNPROCESSED in-domain neighbors.

    Faces on the true X/Y physical boundary are safe to discard because no
    in-domain voxel can lie past them. For H2, outside edges have already
    been inserted at every physical face, and the exterior is a terminal.
    Every Z face stays active up to the slab root, ensuring the two seam
    planes survive. Sibling cross-face witnesses are never discarded.
    """
    outside = int(np.prod(shape, dtype=np.int64))
    ids = graph.ids
    plane = int(shape[1]) * int(shape[2])
    spatial = ids < outside
    z = ids // plane
    y = (ids % plane) // int(shape[2])
    x = ids % int(shape[2])
    terminal = spatial & ((z == box[0][0]) | (z == box[0][1]-1))
    if box[1][0] > 0:
        terminal |= spatial & (y == box[1][0])
    if box[1][1] < shape[1]:
        terminal |= spatial & (y == box[1][1]-1)
    if box[2][0] > 0:
        terminal |= spatial & (x == box[2][0])
    if box[2][1] < shape[2]:
        terminal |= spatial & (x == box[2][1]-1)
    if h2:
        terminal |= ids == outside
    order = stable_voxel_order(graph.levels, h2)
    births, deaths, ea, eb, ew, _ = _graph_sweep(
        ids, graph.values, order, graph.u, graph.v, graph.levels,
        terminal, h2, outside, False)
    sink.write(births, deaths)
    keep = terminal.copy()
    keep[ea] = True
    keep[eb] = True
    selected = np.flatnonzero(keep)
    mapping = np.full(len(ids), -1, np.int32)
    mapping[selected] = np.arange(len(selected), dtype=np.int32)
    return _Graph(ids[selected], graph.values[selected],
                  mapping[ea], mapping[eb], ew)


def _compute_boundary_aware_region(volume, box, shape, offsets, connectivity,
                                   h2, tile_edge, sink, id_dtype, forest_interior=False):
    """Recursive slab tiling with early contraction of true physical sides.

    Unlike the legacy tile recursion, every leaf attaches actual H2 exterior
    contacts before any physical side face can be pruned, and sibling graphs
    share one exterior identifier. The slab root retains two complete Z faces.
    """
    peak_nodes = peak_edges = tiles = 0

    def track(g):
        nonlocal peak_nodes, peak_edges
        peak_nodes = max(peak_nodes, len(g.ids))
        peak_edges = max(peak_edges, len(g.u))

    def visit(region):
        nonlocal tiles
        sizes = [b-a for a,b in region]
        if max(sizes) <= tile_edge:
            tile = np.ascontiguousarray(volume[tuple(slice(a,b) for a,b in region)])
            if tile.dtype.kind == 'f' and not np.isfinite(tile).all():
                raise ValueError('volume contains NaN/Inf')
            graph = _graph_from_leaf(tile, region, shape, offsets, h2, sink, id_dtype)
            if h2:
                graph = _attach_exterior(graph, region[0][0], region[0][1], shape)
            track(graph)
            tiles += 1
        else:
            axis = int(np.argmax(sizes))
            middle = region[axis][0] + sizes[axis] // 2
            lbox, rbox = list(region), list(region)
            lbox[axis] = (region[axis][0], middle)
            rbox[axis] = (middle, region[axis][1])
            left = visit(tuple(lbox))
            right = visit(tuple(rbox))
            if forest_interior and connectivity == 26:
                graph = _merge_internal_forest(left, right, region, axis, middle,
                                               shape, connectivity, h2)
            else:
                graph = _merge_regions_shared(left, right, region, axis, middle,
                                              shape, connectivity, h2)
            del left, right
            track(graph)
        graph = _reduce_physical_sides(graph, region, shape, h2, sink)
        track(graph)
        return graph

    graph = visit(box)
    return graph, peak_nodes, peak_edges, tiles


def _reduce_two_face_slab(graph: _Graph, slab_start: int, slab_end: int,
                          shape: tuple[int, int, int], h2: bool, sink: _Sink) -> _Graph:
    """Discard finished XY side interfaces before joining the rolling prefix.

    Unlike the ordinary region contraction, this slab summary retains ONLY
    its incoming/outgoing Z planes (and the common H2 outside terminal).
    The H2 exterior edges must already have been attached to the whole slab
    before this contraction.  Outside is a shared terminal, not a new birth.
    """
    plane = int(shape[1]) * int(shape[2])
    outside = int(np.prod(shape, dtype=np.int64))
    terminals = ((graph.ids // plane == slab_start) |
                 (graph.ids // plane == slab_end - 1)) & (graph.ids < outside)
    if h2:
        terminals |= graph.ids == outside
    order = stable_voxel_order(graph.levels, h2)
    births, deaths, ea, eb, levels, _ = _graph_sweep(
        graph.ids, graph.values, order, graph.u, graph.v, graph.levels,
        terminals, h2, outside, False)
    sink.write(births, deaths)
    keep = terminals.copy()
    keep[ea] = True
    keep[eb] = True
    selected = np.flatnonzero(keep)
    mapping = np.full(len(graph.ids), -1, np.int32)
    mapping[selected] = np.arange(len(selected), dtype=np.int32)
    return _Graph(graph.ids[selected], graph.values[selected],
                  mapping[ea], mapping[eb], levels)


def _attach_exterior(graph: _Graph, slab_start: int, slab_end: int,
                     shape: tuple[int, int, int]) -> _Graph:
    """Connect *only* newly processed physical boundary voxels to exterior.

    Each boundary voxel appears in its slab's unreduced boundary graph, so
    outside attachments are valid even after recursive in-slab contraction.
    The exterior ID is beyond the maximum voxel ID and remains sorted last.
    """
    z_dim, height, width = shape
    outside = z_dim * height * width
    ids = graph.ids
    plane = height * width
    spatial = ids < outside
    z = ids // plane
    y = (ids % plane) // width
    x = ids % width
    newly_processed = spatial & (z >= slab_start) & (z < slab_end)
    boundary = newly_processed & (
        (x == 0) | (x == width - 1) | (y == 0) | (y == height - 1) |
        (z == 0) | (z == z_dim - 1)
    )
    boundary_nodes = np.flatnonzero(boundary).astype(np.int32, copy=False)
    outside_pos = int(np.searchsorted(ids, outside))
    if outside_pos == len(ids):
        ids = np.concatenate((ids, np.array([outside], dtype=ids.dtype)))
        values = np.concatenate((graph.values, np.zeros(1, graph.values.dtype)))
    else:
        if ids[outside_pos] != outside:
            raise AssertionError('malformed outside ID')
        values = graph.values
    u = np.concatenate((graph.u, boundary_nodes))
    v = np.concatenate((graph.v, np.full(len(boundary_nodes), outside_pos, np.int32)))
    levels = np.concatenate((graph.levels, graph.values[boundary_nodes]))
    return _Graph(ids, values, u, v, levels)


def _reduce_frontier(graph: _Graph, z_end: int, shape, h2: bool,
                     sink: _Sink, final: bool):
    """Retain only the forward plane and global exterior (for H2)."""
    outside = int(np.prod(shape, dtype=np.int64))
    if len(graph.ids) >= np.iinfo(np.int32).max:
        raise MemoryError('frontier graph requires >= 2^31 nodes')
    if final:
        terminals = np.zeros(len(graph.ids), np.bool_)
    else:
        plane = int(shape[1]) * int(shape[2])
        terminals = ((graph.ids // plane) == z_end - 1)
        if h2:
            terminals |= (graph.ids == outside)
    order = stable_voxel_order(graph.levels, h2)
    births, deaths, edges_a, edges_b, weights, essential = _graph_sweep(
        graph.ids, graph.values, order,
        graph.u, graph.v, graph.levels, terminals, h2, outside, final,
    )
    sink.write(births, deaths)
    if final:
        return None, (None if h2 else essential.item() if hasattr(essential, 'item') else essential)
    keep = terminals.copy()
    keep[edges_a] = True
    keep[edges_b] = True
    selected = np.flatnonzero(keep)
    mapping = np.full(len(graph.ids), -1, np.int32)
    mapping[selected] = np.arange(len(selected), dtype=np.int32)
    return _Graph(graph.ids[selected], graph.values[selected],
                  mapping[edges_a], mapping[edges_b], weights), None


def slab_persistence(
    volume, dimension: Literal[0, 2], output_dir: str | Path, *,
    foreground_connectivity: Literal[6, 26] = 6,
    slab_depth: int = 32,
    tile_edge: int = 48,
    write_curve: bool = True,
    compact_ids: bool = True,
    merge_strategy: Literal['forest', 'fused', 'legacy', 'certificate', 'paged-tree', 'stream-parent', 'compact-parent', 'stream26-blocked'] = 'forest',
    intra_slab: Literal['full', 'two-face', 'boundary-aware', 'boundary-forest'] = 'full',
    certificate_memory_mib: float = 128.0,
    event_page_mib: float = 64.0,
    implicit_frontier: bool = False,
    interval_chunk: int = 65536,
    disk_retained: bool = False,
    uf_layout: Literal['separate','interleaved'] = 'separate',
    seam_batch_activations: int = 2048,
    uf_block_entries: int = 65536,
) -> HierarchicalResult:
    """Compute full, exact H0/H2 persistence with a rolling Z slab hierarchy.

    Works with out-of-core and lazily generated 3-D array-like sources. Keeps
    just one reduced prefix boundary graph and one new slab graph. For H2 the
    exterior is attached incrementally to *physical* faces (not artificial
    slab faces). Peak memory is not a hard cap; seam edge allocation may be
    large when a face has very many voxels.
    """
    if njit is None:
        raise ImportError('slab persistence requires numba')
    _validate_source(volume)
    if dimension not in (0, 2):
        raise ValueError('dimension must be 0 or 2')
    if foreground_connectivity not in (6, 26):
        raise ValueError('foreground_connectivity must be 6 or 26')
    if uf_layout not in ('separate','interleaved'):
        raise ValueError('uf_layout must be separate or interleaved')
    if not isinstance(seam_batch_activations, int) or seam_batch_activations < 1:
        raise ValueError('seam_batch_activations must be positive')
    if not isinstance(uf_block_entries, int) or uf_block_entries < 1:
        raise ValueError('uf_block_entries must be positive')
    if not isinstance(interval_chunk, int) or interval_chunk < 1:
        raise ValueError('interval_chunk must be positive')
    if not isinstance(slab_depth, int) or slab_depth < 1:
        raise ValueError('slab_depth must be a positive integer')
    if not isinstance(tile_edge, int) or tile_edge < 2:
        raise ValueError('tile_edge must be at least 2')
    if implicit_frontier and merge_strategy != 'paged-tree':
        raise ValueError('implicit_frontier requires merge_strategy=paged-tree')
    if merge_strategy not in ('forest', 'fused', 'legacy', 'certificate', 'paged-tree', 'stream-parent', 'compact-parent', 'stream26-blocked'):
        raise ValueError('unsupported merge strategy')
    if intra_slab not in ('full', 'two-face', 'boundary-aware', 'boundary-forest'):
        raise ValueError('intra_slab must be full, two-face, boundary-aware or boundary-forest')
    if intra_slab != 'full' and merge_strategy not in ('forest', 'certificate', 'paged-tree', 'stream-parent', 'compact-parent', 'stream26-blocked'):
        raise ValueError('optimized intra-slab contraction requires forest or certificate seam merging')
    if not np.isfinite(event_page_mib) or event_page_mib < 0:
        raise ValueError('event_page_mib must be finite and nonnegative')
    if not np.isfinite(certificate_memory_mib) or certificate_memory_mib < 0:
        raise ValueError('certificate_memory_mib must be nonnegative')
    shape = tuple(map(int, volume.shape))
    h2 = dimension == 2
    connectivity = (26 if foreground_connectivity == 6 else 6) if h2 else foreground_connectivity
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = 'h2' if h2 else 'h0'
    pairs_path = out / f'{key}_pairs.bin'
    metadata_path = out / f'{key}_metadata.json'
    curve_path = out / f'{key}_betti_curve.csv' if write_curve else None
    sink = _Sink(pairs_path, np.dtype(volume.dtype), curve=write_curve)
    merger = (_merge_z_forest if merge_strategy == 'forest' else
              _merge_graphs_lowmem if merge_strategy == 'fused' else _merge_graphs)
    id_dtype = _id_dtype(shape, compact_ids)
    current = None
    peak_nodes = peak_edges = tiles = slabs = 0
    essential = None
    work = tempfile.TemporaryDirectory(prefix='betti_cert_', dir=out) if merge_strategy in ('certificate', 'paged-tree', 'stream-parent', 'compact-parent', 'stream26-blocked') else None
    workspace = CertificateWorkspace(Path(work.name), certificate_memory_mib) if work is not None else None
    def track(graph):
        nonlocal peak_nodes, peak_edges
        peak_nodes = max(peak_nodes, graph.node_count)
        peak_edges = max(peak_edges, graph.edge_count)
    try:
        for z0 in range(0, shape[0], slab_depth):
            z1 = min(z0 + slab_depth, shape[0])
            box = ((z0, z1), (0, shape[1]), (0, shape[2]))
            compute = (_compute_boundary_aware_region if intra_slab in ('boundary-aware', 'boundary-forest')
                       else _compute_region)
            subgraph, pn, pe, ts = compute(
                volume, box, shape, _offsets(connectivity), connectivity, h2,
                tile_edge, sink, id_dtype,
                *(("fused" if merge_strategy in ("forest", "certificate", "paged-tree", "stream-parent") else merge_strategy,)
                  if intra_slab not in ("boundary-aware", "boundary-forest") else
                  (intra_slab == "boundary-forest",)),
            )
            slabs += 1
            tiles += ts
            peak_nodes = max(peak_nodes, pn)
            peak_edges = max(peak_edges, pe)
            if intra_slab == 'two-face':
                if h2:
                    # Attach true physical exterior boundaries WHILE their
                    # voxels remain present. Artificial slab faces stay closed.
                    subgraph = _attach_exterior(subgraph, z0, z1, shape)
                subgraph = _reduce_two_face_slab(subgraph, z0, z1, shape, h2, sink)
                track(subgraph)
            if merge_strategy in ('stream-parent', 'compact-parent', 'stream26-blocked') and h2 and intra_slab == 'full':
                subgraph = _attach_exterior(subgraph, z0, z1, shape)
                track(subgraph)
            last = z1 == shape[0]
            streamed = False
            if current is None:
                current = subgraph
            else:
                preceding = ((0, z0), (0, shape[1]), (0, shape[2]))
                enclosing = ((0, z1), (0, shape[1]), (0, shape[2]))
                if merge_strategy in ('stream-parent', 'compact-parent', 'stream26-blocked'):
                    current, new_essential, stream_stats = stream_parent_reduce(
                        current, subgraph, z1, shape, connectivity, h2, sink, last,
                        workspace, event_page_mib,
                        compact_state=(merge_strategy in ('compact-parent','stream26-blocked')),
                        stream_26=(merge_strategy == 'stream26-blocked'),
                        uf_layout=(uf_layout if merge_strategy == 'stream26-blocked' else 'separate'),
                        seam_batch_activations=seam_batch_activations,
                        uf_block_entries=uf_block_entries,
                        interval_chunk=interval_chunk, disk_retained=disk_retained)
                    peak_nodes = max(peak_nodes, stream_stats['nodes'])
                    peak_edges = max(peak_edges, stream_stats['edges'])
                    streamed = True
                elif merge_strategy == 'paged-tree':
                    if implicit_frontier and isinstance(current, ImplicitFrontier):
                        current = merge_z_implicit(current, subgraph, z0, shape, connectivity, h2, workspace, event_page_mib)
                    else:
                        current = merge_z_paged(current, subgraph, z0, shape, connectivity, h2, workspace, event_page_mib)
                elif merge_strategy == 'certificate':
                    current = merge_z_certificate(current, subgraph, z0, shape, connectivity, h2, workspace)
                elif merge_strategy == 'forest':
                    current = merger(current, subgraph, z0, shape, connectivity, h2)
                else:
                    current = merger(current, subgraph, enclosing, 0, z0,
                                     shape, connectivity, h2)
            if current is not None:
                track(current)
            if h2 and intra_slab == 'full' and merge_strategy not in ('stream-parent', 'compact-parent', 'stream26-blocked'):
                # Only new physical boundary vertices get exterior edges.
                current = _attach_exterior(current, z0, z1, shape)
                track(current)
            if streamed:
                pass  # parent has already been reduced directly from the event runs
            elif merge_strategy == 'paged-tree':
                current, new_essential = reduce_frontier_paged(
                    current, z1, shape, h2, sink, final=last,
                    workspace=workspace, page_memory_mib=event_page_mib,
                    implicit_frontier=implicit_frontier)
            else:
                current, new_essential = _reduce_frontier(
                    current, z1, shape, h2, sink, final=last)
            if workspace is not None:
                # The reduced frontier owns independent dense arrays. Old
                # disk-backed graph buffers are no longer needed.
                workspace.release_merge_files()
            if not last:
                track(current)
            else:
                essential = new_essential
        if sink.deltas is not None and not h2:
            sink.deltas[int(essential)] += 1
        if curve_path is not None and sink.deltas is not None:
            beta = 0
            with curve_path.open('w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(('threshold', 'betti2' if h2 else 'betti0'))
                for level in np.flatnonzero(sink.deltas):
                    beta += int(sink.deltas[level])
                    writer.writerow((int(level), beta))
        elif curve_path is not None:
            sink.handle.flush()
            _write_float_curve_external(pairs_path, curve_path,
                                        dtype=volume.dtype, pair_count=sink.count,
                                        dimension=dimension, essential_birth=essential)
        native_essential = essential.item() if hasattr(essential, 'item') else essential
        metadata = dict(
            algorithm='streaming-z-slab-boundary-contraction', dimensions=shape,
            dtype=str(np.dtype(volume.dtype)), dimension=dimension,
            foreground_connectivity=foreground_connectivity,
            slab_depth=slab_depth, tile_edge=tile_edge, slabs=slabs, tiles=tiles,
            finite_pairs=sink.count, intra_slab=intra_slab, peak_summary_nodes=peak_nodes,
            peak_summary_edges=peak_edges, merge_strategy=merge_strategy,
            compact_ids=compact_ids, global_id_dtype=np.dtype(id_dtype).name,
            implicit_frontier=implicit_frontier,
            certificate_memory_mib=certificate_memory_mib if workspace is not None else None,
            interval_chunk=interval_chunk if merge_strategy in ('compact-parent','stream26-blocked') else None,
            disk_retained=disk_retained if merge_strategy in ('compact-parent','stream26-blocked') else None,
            event_page_mib=event_page_mib if merge_strategy in ('paged-tree', 'stream-parent', 'compact-parent', 'stream26-blocked') else None,
            certificate_mapped_allocations=workspace.mapped_allocations if workspace else 0,
            certificate_peak_backed_bytes=workspace.peak_backed_bytes if workspace else 0,
            certificate_peak_live_disk_bytes=workspace.peak_live_disk_bytes if workspace else 0,
            uf_layout=uf_layout if merge_strategy == 'stream26-blocked' else None,
            seam_batch_activations=seam_batch_activations if merge_strategy == 'stream26-blocked' else None,
            essential_birth=native_essential,
            binary_pair_layout='two consecutive native-endian values per pair',
            curve=str(curve_path) if curve_path is not None else None,
            note='peak graph counts are NOT a peak-RSS or strict memory bound',
        )
        with metadata_path.open('w') as f:
            json.dump(metadata, f, indent=2)
        return HierarchicalResult(pairs_path, metadata_path, curve_path,
                                  sink.count, native_essential, dimension,
                                  str(np.dtype(volume.dtype)), peak_nodes, peak_edges, tiles)
    finally:
        sink.close()
        if work is not None:
            work.cleanup()
