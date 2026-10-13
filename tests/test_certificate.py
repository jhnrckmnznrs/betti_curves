"""Exactness and allocation checks for compact Z-face certificate construction."""
import json
import numpy as np
import pytest

from betti_curves_py import persistence, slab_persistence, load_hierarchical_pairs
from betti_curves_py.certificate import _two_face_integer_order


@pytest.mark.parametrize('dtype', (np.uint8, np.uint16))
@pytest.mark.parametrize('descending', (False, True))
@pytest.mark.parametrize('plane', (1, 13, 250))
def test_two_view_counting_sort_matches_stable(dtype, descending, plane):
    from betti_curves_py.core import stable_voxel_order
    rng = np.random.default_rng(49 + plane + int(descending))
    n = 3*plane + 25
    vals = rng.integers(0, 15, n).astype(dtype)
    got = _two_face_integer_order(vals, 8, plane+15, plane, descending,
                                   256 if dtype == np.uint8 else 65536)
    face = np.r_[vals[8:8+plane], vals[plane+15:2*plane+15]]
    assert np.array_equal(got, stable_voxel_order(face, descending))


@pytest.mark.parametrize('dim', (0, 2))
@pytest.mark.parametrize('conn', (6, 26))
@pytest.mark.parametrize('dtype', (np.uint8, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize('intra', ('full', 'two-face', 'boundary-aware'))
def test_certificate_barcode_exact(tmp_path, dim, conn, dtype, intra):
    rng = np.random.default_rng(111 + dim*200 + conn + int(np.dtype(dtype).itemsize))
    a = rng.integers(0, 17, (8, 9, 11)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        a = a / 7 - 0.7
    oracle = persistence(a, dim, conn, backend='numba-reference')
    result = slab_persistence(a, dim, tmp_path, foreground_connectivity=conn,
                              slab_depth=3, tile_edge=4, write_curve=False,
                              merge_strategy='certificate', intra_slab=intra,
                              certificate_memory_mib=0)
    expected = sorted(zip(oracle.births[:oracle.finite_count],
                          oracle.deaths[:oracle.finite_count]))
    actual = sorted(map(tuple, load_hierarchical_pairs(result).tolist()))
    assert actual == expected
    assert result.essential_birth == (oracle.births[-1] if dim == 0 else None)
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata['certificate_mapped_allocations'] >= 5
    assert metadata['certificate_peak_live_disk_bytes'] > 0
    assert not list(tmp_path.glob('betti_cert_*'))


@pytest.mark.parametrize('conn', (6, 26))
@pytest.mark.parametrize('dim', (0, 2))
def test_certificate_memmapped_volume(tmp_path, dim, conn):
    rng = np.random.default_rng(28 + dim + conn)
    a = rng.integers(0, 65535, (13, 14, 15), dtype=np.uint16)
    p = tmp_path / 'data.npy'
    np.save(p, a)
    src = np.load(p, mmap_mode='r')
    ref = persistence(a, dim, conn, backend='numba-reference')
    res = slab_persistence(src, dim, tmp_path / 'output',
                           foreground_connectivity=conn, slab_depth=5,
                           merge_strategy='certificate', certificate_memory_mib=0,
                           tile_edge=5, write_curve=False)
    assert sorted(map(tuple, load_hierarchical_pairs(res).tolist())) == sorted(
        zip(ref.births[:ref.finite_count],ref.deaths[:ref.finite_count]))


def test_validation(tmp_path):
    a = np.ones((3, 3, 4), np.uint8)
    for bad in (float('nan'), -1, float('inf')):
        with pytest.raises(ValueError):
            slab_persistence(a, 0, tmp_path, merge_strategy='certificate',
                             certificate_memory_mib=bad)

@pytest.mark.parametrize('conn', (6, 26))
@pytest.mark.parametrize('h2', (False, True))
@pytest.mark.parametrize('dtype', (np.uint8, np.uint16, np.float32, np.float64))
def test_cert_forest_identical_edges_even_above_uint32(tmp_path, conn, h2, dtype):
    from betti_curves_py.hierarchical import _Graph
    from betti_curves_py.slab import _merge_z_forest
    from betti_curves_py.certificate import CertificateWorkspace, merge_z_certificate
    shape = (400_000_000, 3, 4)  # virtual shape; never allocate these voxels
    mid = 358_000_000
    plane = shape[1] * shape[2]
    outside = np.prod(shape, dtype=np.int64)
    rng = np.random.default_rng(conn + int(h2) * 11)
    lid = np.arange((mid-1)*plane, mid*plane, dtype=np.int64)
    rid = lid + plane
    lval = rng.integers(1, 21, plane).astype(dtype)
    rval = rng.integers(1, 21, plane).astype(dtype)
    if h2:
        lid = np.r_[lid, outside]
        rid = np.r_[rid, outside]
        lval = np.r_[lval, np.array([0], dtype=dtype)]
        rval = np.r_[rval, np.array([0], dtype=dtype)]
    empty = np.empty(0, np.int32)
    weights = np.empty(0, dtype=dtype)
    left = _Graph(lid, lval, empty, empty, weights)
    right = _Graph(rid, rval, empty, empty, weights)
    reference = _merge_z_forest(left, right, mid, shape, conn, h2)
    workspace = CertificateWorkspace(tmp_path, 0.)
    candidate = merge_z_certificate(left, right, mid, shape, conn, h2, workspace)
    for label in ('ids', 'values', 'u', 'v', 'levels'):
        assert np.array_equal(getattr(candidate, label),getattr(reference, label)),label
    del candidate
    workspace.release_merge_files()
    assert not list(tmp_path.glob('*.bin'))
