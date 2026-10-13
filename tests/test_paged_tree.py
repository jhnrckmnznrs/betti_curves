"""Exactness of disk-paged activation events and elder witness contraction."""
import json
import numpy as np
import pytest

from betti_curves_py import persistence, slab_persistence, load_hierarchical_pairs
from betti_curves_py.hierarchical import _Graph
from betti_curves_py.slab import _merge_z_forest
from betti_curves_py.certificate import CertificateWorkspace
from betti_curves_py.paged import (
    ImplicitZFace, merge_z_paged, _integer_event_order_into,
    _two_face_order_into, reduce_frontier_paged,
)
from betti_curves_py.core import stable_voxel_order


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('desc', [False, True])
@pytest.mark.parametrize('plane', [1, 11, 129])
def test_implicit_event_order(dtype, desc, plane):
    rng = np.random.default_rng(1111 + plane)
    vals = rng.integers(0, 33, size=2*plane+20).astype(dtype)
    actual = np.empty(2*plane, np.int32)
    _two_face_order_into(vals, 7, plane+13, plane, desc,
                         256 if dtype == np.uint8 else 65536, actual)
    ref = stable_voxel_order(np.r_[vals[7:7+plane], vals[plane+13:plane+13+plane]], desc)
    assert np.array_equal(actual, ref)
    edge_order = np.empty(len(vals), np.int32)
    _integer_event_order_into(vals, desc,
                              256 if dtype == np.uint8 else 65536, edge_order)
    assert np.array_equal(edge_order, stable_voxel_order(vals, desc))


def test_virtual_face_ids():
    values = np.zeros(12, np.uint16)
    face = ImplicitZFace(5_000_000_000, 3, 4, values)
    assert face.global_id(11) == 5_000_000_011
    assert np.shares_memory(face.values, values)
    with pytest.raises(IndexError):
        face.global_id(12)


@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('intra', ['full', 'boundary-aware'])
@pytest.mark.parametrize('page', [0.0, 64.0])
def test_paged_exact(tmp_path, dim, conn, dtype, intra, page):
    rng = np.random.default_rng(1029 + dim*17 + conn)
    volume = rng.integers(0, 20, size=(8, 9, 11)).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        volume = volume / 11 - 1.5
    oracle = persistence(volume, dim, conn, backend='numba-reference')
    result = slab_persistence(volume, dim, tmp_path, foreground_connectivity=conn,
                              slab_depth=3, tile_edge=4, write_curve=False,
                              merge_strategy='paged-tree', intra_slab=intra,
                              event_page_mib=page)
    pairs = sorted(map(tuple, load_hierarchical_pairs(result).tolist()))
    expected = sorted(zip(oracle.births[:oracle.finite_count],
                          oracle.deaths[:oracle.finite_count]))
    assert pairs == expected
    assert result.essential_birth == (oracle.births[-1] if dim == 0 else None)
    meta = json.loads(result.metadata_path.read_text())
    if page == 0:
        assert meta['certificate_mapped_allocations'] >= 10
        assert meta['certificate_peak_live_disk_bytes'] > 0
    assert not list(tmp_path.glob('betti_cert_*'))


@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('conn', [6, 26])
def test_paged_memmap_input(tmp_path, dim, conn):
    rng = np.random.default_rng(123 + dim*4 + conn)
    vol = rng.integers(0, 65535, (13, 14, 15), dtype=np.uint16)
    np.save(tmp_path / 'image.npy', vol)
    memmap = np.load(tmp_path / 'image.npy', mmap_mode='r')
    r = slab_persistence(memmap, dim, tmp_path / 'output',
                         foreground_connectivity=conn, slab_depth=4,
                         tile_edge=5, write_curve=True, merge_strategy='paged-tree',
                         intra_slab='boundary-aware', event_page_mib=0)
    p = persistence(vol, dim, conn, backend='numba-reference')
    assert sorted(map(tuple, load_hierarchical_pairs(r))) == sorted(zip(p.births[:p.finite_count],p.deaths[:p.finite_count]))
    assert r.curve_path.exists()


@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint16, np.float32])
def test_paged_merge_array_equality(tmp_path, dim, conn, dtype):
    rng = np.random.default_rng(71 + dim + conn)
    shape = (600_000_000, 4, 5)  # virtual extent; exercises >2**32 voxel IDs
    mid = 500_000_000
    plane = shape[1] * shape[2]
    outside = np.prod(shape, dtype=np.int64)
    l = np.arange((mid-1)*plane, mid*plane, dtype=np.int64)
    r = l + plane
    lv = rng.integers(0, 80, plane).astype(dtype)
    rv = rng.integers(0, 80, plane).astype(dtype)
    if dim == 2:
        l, r = np.r_[l, outside], np.r_[r, outside]
        lv, rv = np.r_[lv, dtype(0)], np.r_[rv, dtype(0)]
    z = np.empty(0, np.int32)
    weights = np.empty(0, dtype)
    left = _Graph(l, lv, z, z, weights)
    right = _Graph(r, rv, z, z, weights)
    ref = _merge_z_forest(left, right, mid, shape, conn, dim == 2)
    ws = CertificateWorkspace(tmp_path, 0.)
    candidate = merge_z_paged(left, right, mid, shape, conn, dim == 2, ws, 0.)
    for attr in ('ids', 'values', 'u', 'v', 'levels'):
        assert np.array_equal(getattr(ref, attr), getattr(candidate, attr)), attr
    del candidate
    ws.release_merge_files()
    assert not list(tmp_path.glob('*.bin'))


def test_paged_validation(tmp_path):
    x = np.ones((2, 3, 4), np.uint16)
    for bad in [-1., np.inf, np.nan]:
        with pytest.raises(ValueError):
            slab_persistence(x, 0, tmp_path, merge_strategy='paged-tree',
                             event_page_mib=bad)
