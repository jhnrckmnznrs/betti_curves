"""Exactness of fused seam assembly, demand-driven scheduling and disk spilling."""
import json

import numpy as np
import pytest

from betti_curves_py import hierarchical_persistence, load_hierarchical_pairs, persistence
from betti_curves_py.hierarchical import _Graph, _merge_graphs, _merge_graphs_lowmem


@pytest.mark.parametrize('shape', [(3, 4, 5), (9, 5, 7)])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dim', [0, 2])
def test_fused_parent_assembly_matches_old_graph(shape, axis, conn, dim):
    random = np.random.default_rng(sum(shape) + axis)
    ids = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    split = shape[axis] // 2
    slices = [slice(None)] * 3
    slices[axis] = slice(None, split)
    lid = np.sort(ids[tuple(slices)].ravel())
    slices[axis] = slice(split, None)
    rid = np.sort(ids[tuple(slices)].ravel())
    def graph(nodes):
        return _Graph(nodes, random.integers(0, 33, size=len(nodes), dtype=np.uint16),
                      np.empty(0, np.int32), np.empty(0, np.int32),
                      np.empty(0, np.uint16))
    l, r = graph(lid), graph(rid)
    box = tuple((0, d) for d in shape)
    expected = _merge_graphs(l, r, box, axis, split, shape, conn, dim == 2)
    actual = _merge_graphs_lowmem(l, r, box, axis, split, shape, conn, dim == 2)
    for field in ('ids', 'values'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    # Direct builder enumerates in a different seam order: compare as edge multisets.
    def edges(g):
        return sorted(zip(g.ids[g.u].tolist(), g.ids[g.v].tolist(), g.levels.tolist()))
    assert edges(actual) == edges(expected)


@pytest.mark.parametrize('dimension', [0, 2])
@pytest.mark.parametrize('connectivity', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
def test_adaptive_and_spilled_intervals_exact(tmp_path, dimension, connectivity, dtype):
    rng = np.random.default_rng(9000 + dimension + connectivity)
    volume = rng.integers(0, 17, size=(12, 13, 14), dtype=np.uint16).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        volume = (volume - 5) * 0.25
    fixed = hierarchical_persistence(
        volume, dimension, tmp_path/'fixed', tile_edge=4,
        foreground_connectivity=connectivity, parallel_workers=3,
        scheduling='fixed')
    adaptive = hierarchical_persistence(
        volume, dimension, tmp_path/'adaptive', tile_edge=4,
        foreground_connectivity=connectivity, parallel_workers=3,
        scheduling='adaptive', task_factor=4, pending_summary_mib=0)
    baseline = persistence(volume, dimension, connectivity, backend='numba-reference')
    expected = sorted(zip(baseline.births[:baseline.finite_count].tolist(),
                          baseline.deaths[:baseline.finite_count].tolist()))
    assert sorted(map(tuple, load_hierarchical_pairs(fixed).tolist())) == expected
    assert sorted(map(tuple, load_hierarchical_pairs(adaptive).tolist())) == expected
    assert adaptive.essential_birth == fixed.essential_birth
    assert adaptive.curve_path.read_bytes() == fixed.curve_path.read_bytes()
    info = json.loads(adaptive.metadata_path.read_text())
    assert info['scheduled_jobs'] > 3
    assert info['adaptive_merges'] == info['scheduled_jobs'] - 1
    assert info['summary_spills'] >= 1
    # Root graph is deliberately kept resident for the final reduction.
    assert info['peak_pending_summary_bytes'] >= 0
    assert not list((tmp_path/'adaptive').glob('betti_workers_*'))


def test_fused_high_unsigned_global_ids():
    # Global linear IDs > 2**31 are supported in compact uint32 mode.
    shape = (2000, 2000, 1000)
    box = ((1200, 1202), (105, 109), (10, 13))
    mid = 1201
    strides = (shape[1] * shape[2], shape[2], 1)
    def node(z, y, x):
        return z*strides[0] + y*strides[1] + x
    left_ids = np.array([node(1200,y,x) for y in range(105,109)
                         for x in range(10,13)], dtype=np.uint32)
    right_ids = np.array([node(1201,y,x) for y in range(105,109)
                          for x in range(10,13)], dtype=np.uint32)
    def graph(ids):
        return _Graph(ids, ids.astype(np.uint16), np.empty(0,np.int32),
                      np.empty(0,np.int32), np.empty(0,np.uint16))
    left,right=graph(left_ids),graph(right_ids)
    for conn in (6,26):
        for h2 in (False,True):
            a=_merge_graphs(left,right,box,0,mid,shape,conn,h2)
            b=_merge_graphs_lowmem(left,right,box,0,mid,shape,conn,h2)
            assert sorted(zip(a.u.tolist(),a.v.tolist(),a.levels.tolist())) == sorted(
                zip(b.u.tolist(),b.v.tolist(),b.levels.tolist()))


def test_adaptive_process_spill(tmp_path):
    v=np.zeros((10,11,12), dtype=np.uint16)
    v[2:8,2:8,2:8] = 10
    fixed=hierarchical_persistence(v, 2, tmp_path/'fixed', tile_edge=3,
                                   parallel_workers=2, scheduling='fixed', write_curve=False)
    adaptive=hierarchical_persistence(v, 2, tmp_path/'adaptive', tile_edge=3,
                                      parallel_workers=2, scheduling='adaptive',
                                      parallel_backend='processes', pending_summary_mib=0,
                                      write_curve=False)
    np.testing.assert_array_equal(np.sort(load_hierarchical_pairs(fixed),axis=0),
                                  np.sort(load_hierarchical_pairs(adaptive),axis=0))
    meta=json.loads(adaptive.metadata_path.read_text())
    assert meta['parallel_input_spooled'] is True
    assert meta['scheduling']=='adaptive'


@pytest.mark.parametrize('kwargs', [
    {'scheduling':'oops'}, {'merge_strategy':'other'},
    {'task_factor':0}, {'pending_summary_mib':-1},
    {'pending_summary_mib':float('nan')}])
def test_new_settings_validation(tmp_path, kwargs):
    with pytest.raises(ValueError):
        hierarchical_persistence(np.ones((2,2,2),dtype=np.uint8), 0,
                                 tmp_path/'bad', **kwargs)
