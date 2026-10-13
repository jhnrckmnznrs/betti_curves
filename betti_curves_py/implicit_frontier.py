"""Experimental exact rolling Z-frontier with affine face terminals.

Only non-face elder witness IDs are stored explicitly. Full forward-face
voxel IDs are reconstructed into the *parent merge output* directly, never
stored in a persistent frontier. This does NOT make the full parent reduction
sublinear in face area: parent graph, edge event order, and union-find state
are still materialized or page-backed during each merge.
"""
from dataclasses import dataclass
import numpy as np

from .hierarchical import _Graph, _jit
from .certificate import CertificateWorkspace
from .paged import _alloc, _two_face_order_into, _paged_forest_into, _init_union_state
from .core import stable_voxel_order


@dataclass
class ImplicitFrontier:
    witness_ids: np.ndarray
    witness_values: np.ndarray
    face_values: np.ndarray
    face_start: int
    outside_id: int | None
    outside_value: object | None
    u: np.ndarray
    v: np.ndarray
    levels: np.ndarray

    @property
    def node_count(self):
        return self.witness_ids.size + self.face_values.size + int(self.outside_id is not None)

    @property
    def edge_count(self):
        return self.u.size

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.witness_ids, self.witness_values,
                   self.face_values, self.u, self.v, self.levels)) + (
                   self.witness_values.dtype.itemsize if self.outside_id is not None else 0)

    def materialize(self):
        """Reference/debug decoder; NOT used by the optimized seam merge."""
        n = self.face_values.size
        ids = np.concatenate((self.witness_ids,
            np.arange(self.face_start, self.face_start+n, dtype=np.int64),
            np.array([self.outside_id], dtype=np.int64) if self.outside_id is not None else np.empty(0, np.int64)))
        values = np.concatenate((self.witness_values, self.face_values,
            np.array([self.outside_value], dtype=self.face_values.dtype) if self.outside_id is not None else np.empty(0, self.face_values.dtype)))
        return _Graph(ids, values, self.u, self.v, self.levels)


def pack_frontier(graph: _Graph, z_end: int, shape, h2: bool) -> ImplicitFrontier:
    """Exact conversion from the existing reduced frontier.

    Enforce complete, row-major forward face; do not guess if the reference
    reducer changed its terminal convention.
    """
    plane = int(shape[1]) * int(shape[2])
    start = (int(z_end) - 1) * plane
    outside = int(np.prod(shape, dtype=np.int64))
    p = int(np.searchsorted(graph.ids, start))
    if p + plane > len(graph.ids) or graph.ids[p] != start or graph.ids[p+plane-1] != start+plane-1:
        raise AssertionError('frontier must contain a complete contiguous forward Z-face')
    if p and np.any(graph.ids[:p] >= start):
        raise AssertionError('witness ID overlaps forward face')
    tail = len(graph.ids) - p - plane
    if h2:
        if tail != 1 or int(graph.ids[-1]) != outside:
            raise AssertionError('H2 frontier must include the unique exterior node')
    elif tail:
        raise AssertionError('H0 frontier unexpectedly contains trailing witnesses')
    if graph.ids.dtype == np.int64:
        # The global range may be smaller than 2^32 but the face may not;
        # preserve signed safe IDs at the actual global position.
        wids = graph.ids[:p].copy()
    else:
        wids = graph.ids[:p].copy()
    return ImplicitFrontier(
        wids, graph.values[:p].copy(), graph.values[p:p+plane].copy(),
        start, outside if h2 else None,
        graph.values[-1].item() if h2 else None,
        graph.u.copy(), graph.v.copy(), graph.levels.copy())


@_jit
def _copy_implicit_nodes(wids, wvals, fvals, face_start, outside, outside_val,
                         right_ids, right_vals, right_has_outside,
                         out_ids, out_vals):
    """Direct sorted parent concatenation; no left IDs or node maps."""
    nw, nf = len(wids), len(fvals)
    nl = nw + nf
    nr = len(right_ids) - int(right_has_outside)
    for i in range(nw):
        out_ids[i] = wids[i]
        out_vals[i] = wvals[i]
    for i in range(nf):
        out_ids[nw+i] = face_start + i
        out_vals[nw+i] = fvals[i]
    for i in range(nr):
        out_ids[nl+i] = right_ids[i]
        out_vals[nl+i] = right_vals[i]
    if outside >= 0:
        out_ids[nl+nr] = outside
        out_vals[nl+nr] = outside_val


@_jit
def _copy_implicit_edges(lu, lv, lw, ru, rv, rw, left_nreal, right_nreal,
                          has_left_outside, has_right_outside, outside_at,
                          out_u, out_v, out_w):
    for i in range(len(lu)):
        a, b = lu[i], lv[i]
        out_u[i] = outside_at if has_left_outside and a == left_nreal else a
        out_v[i] = outside_at if has_left_outside and b == left_nreal else b
        out_w[i] = lw[i]
    for i in range(len(ru)):
        a, b = ru[i], rv[i]
        j = len(lu) + i
        out_u[j] = outside_at if has_right_outside and a == right_nreal else left_nreal + a
        out_v[j] = outside_at if has_right_outside and b == right_nreal else left_nreal + b
        out_w[j] = rw[i]


@_jit
def _direct_matching_seam(values, plane, lb, rb, h2, u, v, levels, offset):
    """6-neighbor Z interface is a matching: no Kruskal events or union-find."""
    for i in range(plane):
        a, b = values[lb+i], values[rb+i]
        u[offset+i] = lb+i
        v[offset+i] = rb+i
        levels[offset+i] = min(a,b) if h2 else max(a,b)
    return offset+plane


def merge_z_implicit(left: ImplicitFrontier, right: _Graph, mid: int, shape,
                     connectivity: int, h2: bool, workspace: CertificateWorkspace,
                     page_memory_mib: float = 64.) -> _Graph:
    """Exact parent graph from implicit Z-face + ordinary next slab graph.

    The preceding prefix and the next slab occupy disjoint ordered global-ID
    ranges, except for their shared H2 exterior. Exploits that fact to avoid
    general node sorting and per-child dense remapping arrays.
    """
    if not np.isfinite(page_memory_mib) or page_memory_mib < 0:
        raise ValueError('page_memory_mib must be finite and nonnegative')
    plane = int(shape[1]) * int(shape[2])
    if left.face_start != (mid-1)*plane or len(left.face_values) != plane:
        raise ValueError('prefix does not end at expected seam face')
    if 2*plane >= np.iinfo(np.int32).max:
        raise MemoryError('seam exceeds local 32-bit indices')
    outside = int(np.prod(shape, dtype=np.int64))
    has_left_outside = left.outside_id is not None
    has_right_outside = bool(h2 and len(right.ids) and int(right.ids[-1]) == outside)
    if has_left_outside and left.outside_id != outside:
        raise ValueError('inconsistent exterior ID')
    if has_left_outside and has_right_outside and left.outside_value != right.values[-1]:
        raise ValueError('inconsistent exterior values')
    nr = len(right.ids) - int(has_right_outside)
    nl = len(left.witness_ids) + plane
    outside_present = has_left_outside or has_right_outside
    nn = nl + nr + int(outside_present)
    edges_cap = len(left.u) + len(right.u) + (plane if connectivity == 6 else 2*plane-1)
    if nn >= np.iinfo(np.int32).max or edges_cap >= np.iinfo(np.int32).max:
        raise MemoryError('parent graph exceeds int32 indices')
    # Ascending disjoint physical slab ranges: no sorted merge necessary.
    if nr and (int(right.ids[0]) < mid*plane or int(right.ids[nr-1]) >= (mid+int(shape[0]))*plane):
        raise AssertionError('unexpected next-slab voxel ID range')
    ids_dtype = np.result_type(left.witness_ids.dtype, right.ids.dtype)
    if int(shape[0])*plane > np.iinfo(np.int32).max:
        ids_dtype = np.dtype('int64')
    limit = page_memory_mib * 1024**2
    ids = _alloc(workspace, 'implicit_parent_ids', nn, ids_dtype, limit)
    vals = _alloc(workspace, 'implicit_parent_values', nn, right.values.dtype, limit)
    outside_value = (left.outside_value if has_left_outside else
                     right.values[-1].item() if has_right_outside else 0)
    _copy_implicit_nodes(left.witness_ids, left.witness_values,
                         left.face_values, left.face_start,
                         outside if outside_present else -1, outside_value,
                         right.ids, right.values, has_right_outside, ids, vals)
    lb = len(left.witness_ids)
    rb = nl + int(np.searchsorted(right.ids[:nr], mid*plane))
    if rb+plane > nl+nr or ids[rb] != mid*plane or ids[rb+plane-1] != (mid+1)*plane-1:
        raise AssertionError('next slab must have complete incoming face')
    u = _alloc(workspace, 'implicit_u', edges_cap, np.int32, limit)
    v = _alloc(workspace, 'implicit_v', edges_cap, np.int32, limit)
    levels = _alloc(workspace, 'implicit_levels', edges_cap, vals.dtype, limit)
    _copy_implicit_edges(left.u, left.v, left.levels,
                         right.u, right.v, right.levels,
                         nl, nr, has_left_outside, has_right_outside,
                         nn-1, u, v, levels)
    start = len(left.u)+len(right.u)
    if connectivity == 6:
        end = _direct_matching_seam(vals, plane, lb, rb, h2,
                                     u, v, levels, start)
        return _Graph(ids, vals, u[:end], v[:end], levels[:end])
    order = _alloc(workspace, 'implicit_activation_order', 2*plane, np.int32, limit)
    if vals.dtype == np.uint8 or vals.dtype == np.uint16:
        _two_face_order_into(vals, lb, rb, plane, h2,
                             256 if vals.dtype == np.uint8 else 65536, order)
    else:
        combined = np.empty(2*plane, dtype=vals.dtype)
        combined[:plane] = vals[lb:lb+plane]
        combined[plane:] = vals[rb:rb+plane]
        order[:] = stable_voxel_order(combined, h2)
        del combined
    parent = _alloc(workspace, 'implicit_parent_uf', 2*plane, np.int32, limit)
    sizes = _alloc(workspace, 'implicit_size_uf', 2*plane, np.int32, limit)
    active = _alloc(workspace, 'implicit_active', 2*plane, np.bool_, limit)
    _init_union_state(parent, sizes, active)
    end = _paged_forest_into(vals, order, plane, int(shape[2]), connectivity,
                             lb, rb, parent, sizes, active, u, v, levels, start)
    return _Graph(ids, vals, u[:end], v[:end], levels[:end])
