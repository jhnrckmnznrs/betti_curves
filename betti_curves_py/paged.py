"""Experimental out-of-core Kruskal event backend for rolling Z boundaries.

Each seam consists of two *implicit* faces indexed by affine global voxel IDs.
The event order, active flags, parent links and union sizes may be paged to disk.
The successful Kruskal edges are stored directly in the parent graph buffer.

This preserves the seam filtered-component functor, but it is NOT a fully
implicit terminal merge tree: the downstream graph still materializes all
boundary terminals and elder witness vertices. It is a measurable step toward
an external-memory implementation, not an asymptotically sublinear interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .hierarchical import _Graph, _find, _jit
from .core import stable_voxel_order
from .certificate import (_merge_sorted_nodes_to, _remap_edges, CertificateWorkspace)


@dataclass(frozen=True)
class ImplicitZFace:
    """A regular Z-face: voxel IDs are start_id + [0, height * width).

    No dense voxel IDs or additional copies of intensities are required.
    ``values`` is a view of a parent graph's already-retained intensities.
    """
    start_id: int
    height: int
    width: int
    values: np.ndarray

    @property
    def size(self):
        return self.height * self.width

    def global_id(self, local_pos):
        if not (0 <= local_pos < self.size):
            raise IndexError(local_pos)
        return self.start_id + int(local_pos)


@_jit
def _two_face_order_into(vals, lb, rb, plane, descending, bins, order):
    """Stable counting sort to caller-provided event queue; no face concatenation.

    The full order can be an on-disk memmap. Stable tie ordering is left-face
    row-major followed by right-face row-major, as in the reference kernel.
    """
    counts = np.zeros(bins, np.int64)
    for i in range(plane):
        counts[int(vals[lb + i])] += 1
        counts[int(vals[rb + i])] += 1
    start = np.empty(bins, np.int64)
    c = 0
    for i in range(bins):
        level = bins - 1 - i if descending else i
        start[level] = c
        c += counts[level]
    for i in range(plane):
        k = int(vals[lb + i])
        j = start[k]
        order[j] = i
        start[k] += 1
    for i in range(plane):
        k = int(vals[rb + i])
        j = start[k]
        order[j] = plane + i
        start[k] += 1


@_jit
def _paged_forest_into(values, activation_order, plane, width, connectivity,
                       left_base, right_base, parent, sizes, active,
                       u, v, levels, start):
    """Online Kruskal event processing with caller-owned union-find arrays."""
    radius = 0 if connectivity == 6 else 1
    height = plane // width
    k = start
    for pos0 in activation_order:
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
                    u[k] = left_base + p
                    v[k] = right_base + yy * width + xx
                    levels[k] = values[left_base + p]
                else:
                    u[k] = left_base + yy * width + xx
                    v[k] = right_base + p
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


def _alloc(workspace, name, count, dtype, threshold_bytes):
    """Allocate mutable event arrays via RAM or disk according to page policy."""
    spill = count * np.dtype(dtype).itemsize >= threshold_bytes
    return workspace.allocate(name, count, dtype, spill)


@_jit
def _init_union_state(parent, sizes, active):
    for i in range(len(parent)):
        parent[i] = i
        sizes[i] = 1
        active[i] = False


def merge_z_paged(left: _Graph, right: _Graph, mid: int, shape,
                  connectivity: int, h2: bool, workspace: CertificateWorkspace,
                  page_memory_mib: float = 64.) -> _Graph:
    """Exact cross-Z forest from implicit face addressing and paged events.

    Caller must reduce the returned graph and release the workspace before
    processing the next slab. Files remain valid while the returned memmaps
    are referenced; no premature deletion or callback-based cleanups.
    """
    if not np.isfinite(page_memory_mib) or page_memory_mib < 0:
        raise ValueError('page_memory_mib must be finite and nonnegative')
    t = page_memory_mib * 1024**2
    plane = int(shape[1]) * int(shape[2])
    if 2 * plane >= np.iinfo(np.int32).max:
        raise MemoryError('seam requires 64-bit *local* event indices')
    outside = int(np.prod(shape, dtype=np.int64))
    shared = bool(h2 and len(left.ids) and len(right.ids)
                  and left.ids[-1] == outside and right.ids[-1] == outside)
    nn = len(left.ids) + len(right.ids) - int(shared)
    max_edges = len(left.u) + len(right.u) + 2 * plane - 1
    if nn >= np.iinfo(np.int32).max or max_edges >= np.iinfo(np.int32).max:
        raise MemoryError('parent graph exceeds 32-bit local indices')
    # The materialized parent remains required by the v0.8 elder sweep.
    # Large output arrays and temporary node maps are paged individually.
    ids = _alloc(workspace, 'node_ids', nn, left.ids.dtype, t)
    vals = _alloc(workspace, 'node_values', nn, left.values.dtype, t)
    lm = _alloc(workspace, 'left_map', len(left.ids), np.int32, t)
    rm = _alloc(workspace, 'right_map', len(right.ids), np.int32, t)
    _merge_sorted_nodes_to(left.ids, right.ids, left.values, right.values,
                           ids, vals, lm, rm, shared)
    lbid, rbid = (mid - 1) * plane, mid * plane
    lb = int(np.searchsorted(ids, lbid))
    rb = int(np.searchsorted(ids, rbid))
    if (lb + plane > nn or rb + plane > nn or ids[lb] != lbid
            or ids[lb + plane - 1] != lbid + plane - 1
            or ids[rb] != rbid or ids[rb + plane - 1] != rbid + plane - 1):
        raise AssertionError('lost complete seam face')
    lf = ImplicitZFace(lbid, shape[1], shape[2], vals[lb:lb+plane])
    rf = ImplicitZFace(rbid, shape[1], shape[2], vals[rb:rb+plane])
    assert lf.size == rf.size == plane
    order = _alloc(workspace, 'events_order', 2*plane, np.int32, t)
    if vals.dtype == np.uint8 or vals.dtype == np.uint16:
        _two_face_order_into(vals, lb, rb, plane, h2,
                             256 if vals.dtype == np.uint8 else 65536, order)
    else:
        # float fallback retains one temporary two-face activation buffer;
        # the output order itself is still disk-backed when requested.
        combined = np.empty(2*plane, vals.dtype)
        combined[:plane] = lf.values
        combined[plane:] = rf.values
        order[:] = stable_voxel_order(combined, h2)
        del combined
    u = _alloc(workspace, 'edges_u', max_edges, np.int32, t)
    v = _alloc(workspace, 'edges_v', max_edges, np.int32, t)
    w = _alloc(workspace, 'edge_levels', max_edges, vals.dtype, t)
    n = _remap_edges(left.u, left.v, left.levels,
                     right.u, right.v, right.levels,
                     lm, rm, u, v, w)
    parent = _alloc(workspace, 'events_parent', 2*plane, np.int32, t)
    sizes = _alloc(workspace, 'events_sizes', 2*plane, np.int32, t)
    active = _alloc(workspace, 'events_active', 2*plane, np.bool_, t)
    _init_union_state(parent, sizes, active)
    end = _paged_forest_into(vals, order, plane, shape[2], connectivity,
                             lb, rb, parent, sizes, active, u, v, w, n)
    workspace.peak_backed_bytes = max(workspace.peak_backed_bytes,
                                      sum(a.nbytes for a in (ids, vals, lm, rm,
                                      order, u, v, w, parent, sizes, active)
                                          if isinstance(a, np.memmap)))
    return _Graph(ids, vals, u[:end], v[:end], w[:end])


@_jit
def _integer_event_order_into(levels, descending, bins, order):
    """Stable full-edge filtration order into possibly disk-backed storage."""
    count = np.zeros(bins, dtype=np.int64)
    for i in range(len(levels)):
        count[int(levels[i])] += 1
    starts = np.empty(bins, np.int64)
    cursor = 0
    for i in range(bins):
        x = bins-1-i if descending else i
        starts[x] = cursor
        cursor += count[x]
    for i in range(len(levels)):
        x = int(levels[i])
        index = starts[x]
        order[index] = i
        starts[x] += 1


@_jit
def _terminal_flags_into(ids, plane, z_end, outside, h2, terminal):
    for i in range(len(ids)):
        terminal[i] = (ids[i] // plane == z_end - 1 or
                       (h2 and ids[i] == outside))


@_jit
def _elder_sweep_into(ids, values, order, edge_a, edge_b, edge_level,
                      terminals, descending, outside_id, final,
                      parent, sizes, elder, touched, witness,
                      births, deaths, kept_a, kept_b, kept_levels):
    """Same elder-rule graph reduction as reference, with paged state arrays.

    Returns event counts rather than copied arrays. A caller may stream finite
    intervals from disk-backed slices directly into the binary output sink.
    """
    n = len(ids)
    for i in range(n):
        parent[i] = i
        sizes[i] = 1
        elder[i] = i
        touched[i] = terminals[i]
        witness[i] = i
    b_count = 0
    e_count = 0
    for kk in order:
        k = int(kk)
        ra = _find(parent, edge_a[k])
        rb = _find(parent, edge_b[k])
        if ra == rb:
            continue
        ea = elder[ra]
        eb = elder[rb]
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
            kept_a[e_count] = old_anchor
            kept_b[e_count] = witness[young_root]
            kept_levels[e_count] = level
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
    essential_birth = values[elder[_find(parent, 0)]]
    return b_count, e_count, essential_birth


@_jit
def _build_mapping(selected, mapping):
    for i in range(len(mapping)):
        mapping[i] = -1
    for i in range(len(selected)):
        mapping[selected[i]] = i


@_jit
def _remap_implicit_edges(keep_a, keep_b, old_witness, face_begin, plane,
                          has_outside, out_a, out_b):
    """Remap kept tree edges without allocating a dense parent-node map."""
    n_witness = len(old_witness)
    for i in range(len(keep_a)):
        for which in range(2):
            old = keep_a[i] if which == 0 else keep_b[i]
            if old < face_begin:
                new = np.searchsorted(old_witness, old)
                if new >= n_witness or old_witness[new] != old:
                    raise AssertionError('missing kept elder witness')
            elif old < face_begin + plane:
                new = n_witness + old - face_begin
            elif has_outside:
                new = n_witness + plane
            else:
                raise AssertionError('terminal ID outside affine face')
            if which == 0:
                out_a[i] = new
            else:
                out_b[i] = new


def reduce_frontier_paged(graph: _Graph, z_end: int, shape, h2: bool,
                          sink, final: bool, workspace: CertificateWorkspace,
                          page_memory_mib: float, implicit_frontier: bool = False):
    """External-page-capable persistence elder sweep and witness contraction.

    Handles complete finite intervals and the outside elder exactly, using
    integer stable counting-order events or a floating-point stable fallback.
    The final retained graph is currently copied into compact dense arrays;
    this is the remaining explicit-terminal boundary bottleneck.
    """
    threshold = page_memory_mib * 1024**2
    n, ne = len(graph.ids), len(graph.u)
    if n >= np.iinfo(np.int32).max or ne >= np.iinfo(np.int32).max:
        raise MemoryError('paged elder sweep needs 64-bit *local* indices')
    outside = int(np.prod(shape, dtype=np.int64))
    terminal = _alloc(workspace, 'terminal_flags', n, np.bool_, threshold)
    if final:
        terminal[:] = False
    else:
        _terminal_flags_into(graph.ids, int(shape[1])*int(shape[2]), z_end,
                             outside, h2, terminal)
    order = _alloc(workspace, 'edge_event_order', ne, np.int32, threshold)
    if graph.levels.dtype == np.uint8 or graph.levels.dtype == np.uint16:
        _integer_event_order_into(graph.levels, h2,
            256 if graph.levels.dtype == np.uint8 else 65536, order)
    else:
        order[:] = stable_voxel_order(graph.levels, h2)
    parent = _alloc(workspace, 'elder_parent', n, np.int32, threshold)
    sizes = _alloc(workspace, 'elder_size', n, np.int32, threshold)
    elder = _alloc(workspace, 'elder_elder', n, np.int32, threshold)
    touched = _alloc(workspace, 'elder_touched', n, np.bool_, threshold)
    witness = _alloc(workspace, 'elder_witness', n, np.int32, threshold)
    births = _alloc(workspace, 'born', n, graph.values.dtype, threshold)
    deaths = _alloc(workspace, 'died', n, graph.values.dtype, threshold)
    kept_a = _alloc(workspace, 'keep_a', n, np.int32, threshold)
    kept_b = _alloc(workspace, 'keep_b', n, np.int32, threshold)
    kept_level = _alloc(workspace, 'keep_level', n, graph.values.dtype, threshold)
    bcount, ecount, essential = _elder_sweep_into(
        graph.ids, graph.values, order, graph.u, graph.v,
        graph.levels, terminal, h2, outside, final, parent, sizes,
        elder, touched, witness, births, deaths, kept_a, kept_b, kept_level)
    sink.write(births[:bcount], deaths[:bcount])
    if final:
        return None, (None if h2 else essential.item() if hasattr(essential,'item') else essential)
    terminal[kept_a[:ecount]] = True
    terminal[kept_b[:ecount]] = True
    if implicit_frontier:
        # Contract the next interface without a full selected-ID array or
        # O(parent-nodes) remapping table. The only dictionary is the sorted
        # list of sparse, pre-face witness indices.
        from .implicit_frontier import ImplicitFrontier
        plane = int(shape[1]) * int(shape[2])
        face_start = (int(z_end)-1) * plane
        first = int(np.searchsorted(graph.ids, face_start))
        if first+plane > n or (int(graph.ids[first]) != face_start or
                int(graph.ids[first+plane-1]) != face_start+plane-1):
            raise AssertionError('incomplete forward Z-face')
        if not np.all(terminal[first:first+plane]):
            raise AssertionError('forward face terminals were discarded')
        if h2:
            if int(graph.ids[-1]) != outside or not terminal[-1]:
                raise AssertionError('missing H2 exterior')
        # No ordinary witness can lie after the whole forward face.
        if np.any(terminal[first+plane:n-int(h2)]):
            raise AssertionError('unexpected trailing witness')
        old_witness = np.flatnonzero(terminal[:first]).astype(np.int32)
        mapped_a = np.empty(ecount, np.int32)
        mapped_b = np.empty(ecount, np.int32)
        _remap_implicit_edges(kept_a[:ecount], kept_b[:ecount], old_witness,
                              first, plane, h2, mapped_a, mapped_b)
        return ImplicitFrontier(
            graph.ids[old_witness].copy(), graph.values[old_witness].copy(),
            graph.values[first:first+plane].copy(), face_start,
            outside if h2 else None,
            graph.values[-1].item() if h2 else None,
            mapped_a, mapped_b, kept_level[:ecount].copy()), None
    selected = np.flatnonzero(terminal)
    mapping = _alloc(workspace, 'elder_compact_map', n, np.int32, threshold)
    _build_mapping(selected, mapping)
    # Materialize the surviving terminal certificate independently of the
    # workspace. Once this returns, temporary paging files may be unlinked.
    return _Graph(np.asarray(graph.ids[selected]),
                  np.asarray(graph.values[selected]),
                  mapping[kept_a[:ecount]].copy(),
                  mapping[kept_b[:ecount]].copy(),
                  kept_level[:ecount].copy()), None
