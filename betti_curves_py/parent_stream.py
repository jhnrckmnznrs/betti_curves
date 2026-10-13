"""Experimental *fused* parent reduction for six-neighbor Z seams.

The parent graph is NEVER assembled.  Instead, three sorted event streams
(left summary, right summary, and the opposite-face matching) are merged in a
single Numba kernel.  Node IDs and intensities are looked up in disjoint child
arrays, with a single shared H2 exterior.  Intervals are finalized immediately
and only the next forward-face certificate is materialized.

The implementation is exact for ascending H0 and descending (dual) H2; it is
not yet an external-memory bounded-RSS implementation.  Union-find and the
retained forward-face must still fit in memory or in the configured memmaps.
"""
from __future__ import annotations

import numpy as np

from .hierarchical import _Graph, _find, _jit
from .core import stable_voxel_order
from .paged import _alloc
from .seam26 import generate_seam26_batch


@_jit
def _node_id(pos, left_ids, right_ids, nl, nr, shared):
    if pos < nl:
        return left_ids[pos]
    if pos < nl + nr:
        return right_ids[pos - nl]
    return left_ids[len(left_ids)-1]  # unique H2 exterior


@_jit
def _node_value(pos, left_values, right_values, nl, nr, shared):
    if pos < nl:
        return left_values[pos]
    if pos < nl + nr:
        return right_values[pos-nl]
    return left_values[len(left_values)-1]


@_jit
def _sweep_three_sorted_runs(
    left_ids, left_values, lu, lv, lw, lo,
    right_ids, right_values, ru, rv, rw, ro,
    seam_order, seam_u, seam_v, seam_weights, seam_matching, left_face, right_face, nl, nr, shared, seam_count,
    face_start, face_end, outside_id, descending, final,
    parent, sizes, elder, touched, witness,
    births, deaths, kept_a, kept_b, kept_weight,
):
    """Kruskal merging three runs, tie order left -> right -> seam."""
    n = nl + nr + int(shared)
    for i in range(n):
        parent[i] = i
        sizes[i] = 1
        elder[i] = i
        wid = _node_id(i, left_ids, right_ids, nl, nr, shared)
        touched[i] = not final and (face_start <= wid < face_end or
                                     (shared and wid == outside_id))
        witness[i] = i
    il = ir = is_ = 0
    count_birth = count_kept = 0
    while il < len(lw) or ir < len(rw) or is_ < seam_count:
        # 3-way stable merge; use child stream ordering for equal values.
        src = -1
        if il < len(lw):
            src = 0
            level = lw[il if len(lo) == 0 else lo[il]]
        if ir < len(rw):
            new = rw[ir if len(ro) == 0 else ro[ir]]
            if src < 0 or (new > level if descending else new < level):
                src = 1
                level = new
        if is_ < seam_count:
            pos = int(is_ if len(seam_order) == 0 else seam_order[is_])
            if seam_matching:
                a = left_values[left_face + pos]
                b = right_values[right_face + pos]
                new = (min(a,b) if descending else max(a,b))
            else:
                new = seam_weights[pos]
            if src < 0 or (new > level if descending else new < level):
                src = 2
                level = new
        if src == 0:
            ix = int(il if len(lo) == 0 else lo[il]); il += 1
            aa = int(lu[ix]); bb = int(lv[ix])
            aa = nl + nr if shared and aa == nl else aa
            bb = nl + nr if shared and bb == nl else bb
        elif src == 1:
            ix = int(ir if len(ro) == 0 else ro[ir]); ir += 1
            aa = int(ru[ix]); bb = int(rv[ix])
            aa = nl + nr if shared and aa == nr else nl + aa
            bb = nl + nr if shared and bb == nr else nl + bb
        else:
            pos = int(is_ if len(seam_order) == 0 else seam_order[is_]); is_ += 1
            if seam_matching:
                aa = left_face + pos
                bb = nl + right_face + pos
            else:
                aa = seam_u[pos]
                bb = seam_v[pos]
        ra = _find(parent, aa)
        rb = _find(parent, bb)
        if ra == rb:
            continue
        ea = elder[ra]
        eb = elder[rb]
        a_id = _node_id(ea, left_ids, right_ids, nl, nr, shared)
        b_id = _node_id(eb, left_ids, right_ids, nl, nr, shared)
        if descending and (a_id == outside_id or b_id == outside_id):
            a_older = a_id == outside_id
        else:
            av = _node_value(ea, left_values, right_values, nl, nr, shared)
            bv = _node_value(eb, left_values, right_values, nl, nr, shared)
            a_older = (av > bv or (av == bv and a_id < b_id)) if descending else (av < bv or (av == bv and a_id < b_id))
        oldroot = ra if a_older else rb
        youngroot = rb if a_older else ra
        oldelder = elder[oldroot]
        youngelder = elder[youngroot]
        old_terminal = touched[oldroot]
        young_terminal = touched[youngroot]
        old_anchor = witness[oldroot] if old_terminal else oldelder
        if not final and young_terminal:
            kept_a[count_kept] = old_anchor
            kept_b[count_kept] = witness[youngroot]
            kept_weight[count_kept] = level
            count_kept += 1
        elif _node_value(youngelder, left_values, right_values, nl, nr, shared) != level:
            b = _node_value(youngelder, left_values, right_values, nl, nr, shared)
            births[count_birth] = level if descending else b
            deaths[count_birth] = b if descending else level
            count_birth += 1
        if sizes[ra] < sizes[rb]:
            newroot, loser = rb, ra
        else:
            newroot, loser = ra, rb
        parent[loser] = newroot
        sizes[newroot] += sizes[loser]
        elder[newroot] = oldelder
        touched[newroot] = old_terminal or young_terminal
        witness[newroot] = old_anchor if old_terminal or young_terminal else -1
    essential = _node_value(elder[_find(parent, 0)], left_values, right_values, nl, nr, shared)
    return count_birth, count_kept, essential


def _seam_levels(left_vals, right_vals, h2):
    return np.minimum(left_vals, right_vals) if h2 else np.maximum(left_vals, right_vals)


def stream_parent_reduce(left: _Graph, right: _Graph, z_end: int, shape,
                         connectivity: int, h2: bool, sink, final: bool,
                         workspace, page_memory_mib: float = 128., *,
                         compact_state: bool = False, interval_chunk: int = 65536,
                         disk_retained: bool = False, stream_26: bool = False,
                         uf_layout: str = "separate", seam_batch_activations: int = 2048,
                         uf_block_entries: int = 65536):
    """Merge and reduce a Z parent directly, never forming a parent `_Graph`.

    Returns ``(next_graph, essential_birth, stats)``. This version supports
    six-neighbor adjacency *between* Z slabs. For H2 this corresponds to
    26-connected foreground and its six-connected background.
    """
    if not isinstance(interval_chunk, int) or interval_chunk < 1:
        raise ValueError('interval_chunk must be positive')
    if uf_layout not in ('separate', 'interleaved'):
        raise ValueError('uf_layout must be separate or interleaved')
    if not isinstance(seam_batch_activations, int) or seam_batch_activations < 1:
        raise ValueError('seam_batch_activations must be positive')
    if not isinstance(uf_block_entries, int) or uf_block_entries < 1:
        raise ValueError('uf_block_entries must be positive')
    if stream_26 and not compact_state:
        raise ValueError('streaming seam requires the compact reducer')
    if connectivity not in (6, 26):
        raise ValueError('unsupported connectivity')
    if not np.isfinite(page_memory_mib) or page_memory_mib < 0:
        raise ValueError('page_memory_mib must be finite and nonnegative')
    plane = int(shape[1])*int(shape[2])
    outside_id = int(np.prod(shape, dtype=np.int64))
    shared = bool(h2)
    if shared and (int(left.ids[-1]) != outside_id or int(right.ids[-1]) != outside_id):
        raise ValueError('H2 children must each contain the common exterior')
    nl = len(left.ids)-int(shared)
    nr = len(right.ids)-int(shared)
    n = nl+nr+int(shared)
    if n >= np.iinfo(np.int32).max:
        raise MemoryError('parent local node indices exceed int32 capacity')
    # Slab start is the first right voxel's Z coordinate (elder witnesses
    # cannot precede the right slab's own physical start).
    mid = int(right.ids[0]) // plane
    lp = int(np.searchsorted(left.ids[:nl], (mid-1)*plane))
    rp = int(np.searchsorted(right.ids[:nr], mid*plane))
    if (lp+plane > nl or rp+plane > nr or
        int(left.ids[lp]) != (mid-1)*plane or
        int(left.ids[lp+plane-1]) != mid*plane-1 or
        int(right.ids[rp]) != mid*plane or
        int(right.ids[rp+plane-1]) != (mid+1)*plane-1):
        raise AssertionError('missing complete seam face')
    if nr and (int(right.ids[0]) < mid*plane or int(left.ids[nl-1]) >= mid*plane):
        raise AssertionError('overlapping child voxel ranges')
    # Sorted child event runs do not require a parent-level edge list or sort.
    # The stable fallback handles arbitrary child graph ordering.
    lo = np.empty(0, dtype=np.int32) if (len(left.levels) < 2 or
        np.all(left.levels[:-1] >= left.levels[1:] if h2 else left.levels[:-1] <= left.levels[1:])) else stable_voxel_order(left.levels, h2).astype(np.int32)
    ro = np.empty(0, dtype=np.int32) if (len(right.levels) < 2 or
        np.all(right.levels[:-1] >= right.levels[1:] if h2 else right.levels[:-1] <= right.levels[1:])) else stable_voxel_order(right.levels, h2).astype(np.int32)
    left_face_values = left.values[lp:lp+plane]
    right_face_values = right.values[rp:rp+plane]
    if connectivity == 6:
        # A matching: one edge per pixel, no seam Kruskal or adjacency graph.
        seam_levels = _seam_levels(left_face_values, right_face_values, h2)
        so = stable_voxel_order(seam_levels, h2).astype(np.int32)
        su = np.empty(0, np.int32)
        sv = np.empty(0, np.int32)
        sw = np.empty(0, dtype=left.values.dtype)
        seam_matching = True
    elif stream_26:
        # Independent activation orders, not a materialized seam forest.
        # The two orders are merged inside generate_seam26_batch.
        so = np.empty(0, np.int32)
        su = np.empty(0, np.int32)
        sv = np.empty(0, np.int32)
        sw = np.empty(0, dtype=left.values.dtype)
        seam_matching = False
        face_left_order = stable_voxel_order(left_face_values, h2).astype(np.int32)
        face_right_order = stable_voxel_order(right_face_values, h2).astype(np.int32)
    else:
        # 26-neighbor seam requires a bipartite filtration-preserving forest,
        # but it too is fed directly to the parent reducer, never assembled
        # into a combined parent graph.
        from .slab import _seam_spanning_forest
        face_values = np.concatenate((left_face_values, right_face_values))
        face_order = stable_voxel_order(face_values, h2)
        su, sv, sw = _seam_spanning_forest(face_values, face_order,
                                           int(shape[2]), connectivity, h2,
                                           lp, nl+rp)
        del face_values, face_order
        so = (np.arange(len(sw), dtype=np.int32) if len(sw) < 2 or
              np.all(sw[:-1] >= sw[1:] if h2 else sw[:-1] <= sw[1:])
              else stable_voxel_order(sw, h2).astype(np.int32))
        seam_matching = False
    seam_events_total = len(so)
    limit = page_memory_mib * 1024**2
    def alloc(key, dt):
        return _alloc(workspace, 'stream_'+key, n, dt, limit)
    if compact_state and uf_layout == 'interleaved':
        # Root/elder/witness are adjacent in a 12-byte record. Initialize
        # one fixed-size block at a time, without an n-sized arange temporary.
        state = _alloc(workspace, 'stream_blocked_state', 3*n, np.int32, limit).reshape(n, 3)
        parent, elder, witness = state[:, 0], state[:, 1], state[:, 2]
    else:
        parent = alloc('parent', np.int32)
        elder = alloc('elder', np.int32)
    if compact_state:
        # Negative signed-parent root entry encodes component size. Terminal
        # status is witness>=0, eliminating both sizes and touched arrays.
        if uf_layout == 'separate':
            witness = alloc('witness', np.int32)
        for b0 in range(0, n, uf_block_entries):
            b1 = min(n, b0 + uf_block_entries)
            parent[b0:b1] = -1
            elder[b0:b1] = np.arange(b0, b1, dtype=np.int32)
            witness[b0:b1] = -1
        fstart, fend = (z_end - 1)*plane, z_end*plane
        begin = int(np.searchsorted(right.ids[:nr], fstart))
        end = int(np.searchsorted(right.ids[:nr], fend))
        witness[nl+begin:nl+end] = np.arange(nl+begin, nl+end, dtype=np.int32)
        if shared:
            witness[-1] = n-1
        # With direct streaming, 'so' is intentionally empty even though
        # seam events will follow. Avoid a one-event output chunk.
        kcap = (interval_chunk if stream_26 and connectivity == 26 else
                min(interval_chunk, max(1, len(left.levels)+len(right.levels)+len(so))))
        births = np.empty(kcap, dtype=left.values.dtype)
        deaths = np.empty(kcap, dtype=left.values.dtype)
        ka = np.empty(kcap, dtype=np.int32)
        kb = np.empty(kcap, dtype=np.int32)
        kw = np.empty(kcap, dtype=left.values.dtype)
        # The retained edge count, unlike the union state, can be much
        # smaller than n. Append exactly as many records as survive.
        keep_path = (workspace.path / 'retained_stream_records.bin') if disk_retained else None
        retained = _RetainedSegments(left.values.dtype, keep_path)
        il = ir = is_ = cb = 0
        seam_events_total = 0
        try:
            def flush_chunk(order, eu, ev, ew, seam_count, halt):
                nonlocal il, ir, cb, seam_events_total
                sj = 0
                while il < len(left.levels) or ir < len(right.levels) or sj < seam_count:
                    if halt and sj >= seam_count:
                        break
                    old = (il, ir, sj)
                    il, ir, sj, nb, ne = _sweep_three_runs_compact_chunk(
                        left.ids, left.values, left.u, left.v, left.levels, lo,
                        right.ids, right.values, right.u, right.v, right.levels, ro,
                        order, eu, ev, ew, seam_matching, lp, rp, nl, nr, shared, seam_count,
                        fstart, fend, outside_id, h2, final,
                        parent, elder, witness, births, deaths, ka, kb, kw,
                        il, ir, sj, kcap, halt, stream_26)
                    if old == (il, ir, sj):
                        raise RuntimeError('compact event sweep did not advance')
                    sink.write(births[:nb], deaths[:nb])
                    cb += nb
                    if not final:
                        retained.append(ka[:ne], kb[:ne], kw[:ne])
                return sj

            if stream_26 and connectivity == 26:
                # Recycle at most nine candidate edges for each activated face
                # pixel. No seam forest, 2-face values, or 9*plane edge array.
                cap = 9 * seam_batch_activations
                eu = np.empty(cap, dtype=np.int32)
                ev = np.empty(cap, dtype=np.int32)
                ew = np.empty(cap, dtype=left.values.dtype)
                empty_order = np.empty(0, dtype=np.int32)
                pl = pr = 0
                while pl < plane or pr < plane:
                    pl, pr, count = generate_seam26_batch(
                        left_face_values, right_face_values, face_left_order, face_right_order,
                        lp, rp, nl, int(shape[2]), h2, pl, pr,
                        seam_batch_activations, eu, ev, ew)
                    seam_events_total += count
                    if count:
                        flushed = flush_chunk(empty_order, eu, ev, ew, count, True)
                        if flushed != count:
                            raise RuntimeError('incomplete seam event buffer')
                flush_chunk(empty_order, eu[:0], ev[:0], ew[:0], 0, False)
            else:
                flush_chunk(so, su, sv, sw, len(so), False)
                seam_events_total = len(so)
            records = retained.finish()
        finally:
            retained.close()
        ce = len(records)
        kept_a, kept_b, kept_w = records['a'], records['b'], records['level']
        root = _find_signed(parent, 0)
        e = elder[root]
        essential = _node_value(e, left.values, right.values, nl, nr, shared)
    else:
        sizes = alloc('size', np.int32)
        touched = alloc('touched', np.bool_)
        witness = alloc('witness', np.int32)
        births = alloc('births', left.values.dtype)
        deaths = alloc('deaths', left.values.dtype)
        kept_a = alloc('kept_a', np.int32)
        kept_b = alloc('kept_b', np.int32)
        kept_w = alloc('kept_weight', left.values.dtype)
        cb, ce, essential = _sweep_three_sorted_runs(
            left.ids, left.values, left.u, left.v, left.levels, lo,
            right.ids, right.values, right.u, right.v, right.levels, ro,
            so, su, sv, sw, seam_matching, lp, rp, nl, nr, shared, len(so),
            (z_end-1)*plane, z_end*plane, outside_id, h2, final,
            parent, sizes, elder, touched, witness,
            births, deaths, kept_a, kept_b, kept_w)
        sink.write(births[:cb], deaths[:cb])
    stats = {'nodes': n, 'edges': len(left.u)+len(right.u)+seam_events_total,
             'seam_events': seam_events_total, 'retained_edges': int(ce),
             'pair_count_new': int(cb)}
    if final:
        return None, (None if h2 else essential.item() if hasattr(essential, 'item') else essential), stats
    # Select forward-face terminals, shared exterior, and retained elder seeds.
    keep = np.zeros(n, dtype=np.bool_)
    if compact_state:
        # The forward face is a sorted contiguous ID range, so its terminal
        # marks need neither a per-node Python loop nor a temporary mask.
        keep[nl+begin:nl+end] = True
    else:
        for j in range(nl, nl+nr):
            gid = int(right.ids[j-nl]); keep[j] = (z_end-1)*plane <= gid < z_end*plane
    if shared:
        keep[-1] = True
    keep[kept_a[:ce]] = True
    keep[kept_b[:ce]] = True
    selected = np.flatnonzero(keep).astype(np.int32)
    # Unlike a dense parent graph, only surviving vertices are decoded.
    lsel = selected < nl
    rsel = (selected >= nl) & (selected < nl+nr)
    out_ids = np.empty(len(selected), dtype=np.result_type(left.ids.dtype, right.ids.dtype))
    out_vals = np.empty(len(selected), dtype=left.values.dtype)
    out_ids[lsel] = left.ids[selected[lsel]]
    out_vals[lsel] = left.values[selected[lsel]]
    out_ids[rsel] = right.ids[selected[rsel]-nl]
    out_vals[rsel] = right.values[selected[rsel]-nl]
    if shared:
        out_ids[-1] = outside_id
        out_vals[-1] = left.values[-1]
    return _Graph(out_ids, out_vals,
                  np.searchsorted(selected, kept_a[:ce]).astype(np.int32),
                  np.searchsorted(selected, kept_b[:ce]).astype(np.int32),
                  np.asarray(kept_w[:ce]).copy()), None, stats

@_jit
def _find_signed(parent, x):
    """Path halving on an int32 parent array whose negative roots hold sizes."""
    while parent[x] >= 0:
        p = parent[x]
        if parent[p] >= 0:
            parent[x] = parent[p]
        x = p
    return x


@_jit
def _sweep_three_runs_compact_chunk(
    left_ids, left_values, lu, lv, lw, lo,
    right_ids, right_values, ru, rv, rw, ro,
    seam_order, seam_u, seam_v, seam_weights, seam_matching,
    left_face, right_face, nl, nr, shared, seam_count,
    face_start, face_end, outside_id, descending, final,
    parent, elder, witness,
    births, deaths, kept_a, kept_b, kept_weight,
    il, ir, is_, max_events, stop_at_seam_end=False, cache_seam_roots=False,
):
    """Process at most max_events sorted events, emitting at most that many records.

    ``parent`` stores negative component sizes at roots; ``witness`` uses -1
    to encode a component with no forward terminal. The Numba state persists
    across invocations, while all three output buffers are reusable.
    """
    count_birth = count_kept = processed = 0
    last_a = last_b = -1
    last_ra = last_rb = -1
    while (processed < max_events and
           (il < len(lw) or ir < len(rw) or is_ < seam_count)):
        if stop_at_seam_end and is_ >= seam_count:
            break
        src = -1
        level = left_values[0]  # overwritten whenever any event is available
        if il < len(lw):
            src = 0
            level = lw[il if len(lo) == 0 else lo[il]]
        if ir < len(rw):
            new = rw[ir if len(ro) == 0 else ro[ir]]
            if src < 0 or (new > level if descending else new < level):
                src = 1
                level = new
        if is_ < seam_count:
            pos = int(is_ if len(seam_order) == 0 else seam_order[is_])
            if seam_matching:
                a = left_values[left_face + pos]
                b = right_values[right_face + pos]
                new = (min(a, b) if descending else max(a, b))
            else:
                new = seam_weights[pos]
            if src < 0 or (new > level if descending else new < level):
                src = 2
                level = new
        processed += 1
        if src == 0:
            ix = int(il if len(lo) == 0 else lo[il]); il += 1
            aa = int(lu[ix]); bb = int(lv[ix])
            aa = nl + nr if shared and aa == nl else aa
            bb = nl + nr if shared and bb == nl else bb
        elif src == 1:
            ix = int(ir if len(ro) == 0 else ro[ir]); ir += 1
            aa = int(ru[ix]); bb = int(rv[ix])
            aa = nl + nr if shared and aa == nr else nl + aa
            bb = nl + nr if shared and bb == nr else nl + bb
        else:
            pos = int(is_ if len(seam_order) == 0 else seam_order[is_]); is_ += 1
            if seam_matching:
                aa = left_face + pos
                bb = nl + right_face + pos
            else:
                aa = seam_u[pos]
                bb = seam_v[pos]
        if cache_seam_roots and src == 2:
            ra = last_ra if aa == last_a else _find_signed(parent, aa)
            rb = last_rb if bb == last_b else _find_signed(parent, bb)
        else:
            ra = _find_signed(parent, aa)
            rb = _find_signed(parent, bb)
            # Child events can change cached seam component roots.
            last_a = last_b = -1
        if ra == rb:
            if cache_seam_roots and src == 2:
                last_a, last_b = aa, bb
                last_ra = last_rb = ra
            continue
        ea, eb = elder[ra], elder[rb]
        a_id = _node_id(ea, left_ids, right_ids, nl, nr, shared)
        b_id = _node_id(eb, left_ids, right_ids, nl, nr, shared)
        if descending and (a_id == outside_id or b_id == outside_id):
            a_older = a_id == outside_id
        else:
            av = _node_value(ea, left_values, right_values, nl, nr, shared)
            bv = _node_value(eb, left_values, right_values, nl, nr, shared)
            a_older = (av > bv or (av == bv and a_id < b_id)) if descending else (av < bv or (av == bv and a_id < b_id))
        oldroot = ra if a_older else rb
        youngroot = rb if a_older else ra
        oldelder = elder[oldroot]
        youngelder = elder[youngroot]
        old_terminal = witness[oldroot] >= 0
        young_terminal = witness[youngroot] >= 0
        old_anchor = witness[oldroot] if old_terminal else oldelder
        if not final and young_terminal:
            kept_a[count_kept] = old_anchor
            kept_b[count_kept] = witness[youngroot]
            kept_weight[count_kept] = level
            count_kept += 1
        elif _node_value(youngelder, left_values, right_values, nl, nr, shared) != level:
            b = _node_value(youngelder, left_values, right_values, nl, nr, shared)
            births[count_birth] = level if descending else b
            deaths[count_birth] = b if descending else level
            count_birth += 1
        # Negative parent values combine size and root marker. Union by size.
        # On equal sizes choose ra, matching the original kernel's tie rule.
        if parent[ra] > parent[rb]:
            newroot, loser = rb, ra
        else:
            newroot, loser = ra, rb
        parent[newroot] += parent[loser]
        parent[loser] = newroot
        elder[newroot] = oldelder
        witness[newroot] = old_anchor if old_terminal or young_terminal else -1
        if cache_seam_roots and src == 2:
            last_a, last_b = aa, bb
            last_ra = last_rb = newroot
    return il, ir, is_, count_birth, count_kept


class _RetainedSegments:
    """Append-only edge segments; can spill records to scratch instead of RAM.

    Segments are kept in aligned record form so no full-size three-array
    reservation is made before knowing how many edges survive the reduction.
    """
    def __init__(self, dtype, path=None):
        self.dtype = np.dtype([('a', '<i4'), ('b', '<i4'), ('level', np.dtype(dtype))], align=False)
        self.path = path
        self.parts = []
        self.count = 0
        self.handle = open(path, 'wb') if path is not None else None

    def append(self, a, b, level):
        if not len(a):
            return
        arr = np.empty(len(a), dtype=self.dtype)
        arr['a'], arr['b'], arr['level'] = a, b, level
        if self.handle is None:
            self.parts.append(arr)
        else:
            arr.tofile(self.handle)
        self.count += len(a)

    def finish(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            if self.count == 0:
                return np.empty(0, dtype=self.dtype)
            return np.memmap(self.path, dtype=self.dtype, mode='r', shape=(self.count,))
        if not self.parts:
            return np.empty(0, dtype=self.dtype)
        if len(self.parts) == 1:
            return self.parts[0]
        return np.concatenate(self.parts)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
