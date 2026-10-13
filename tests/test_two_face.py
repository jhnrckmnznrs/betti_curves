"""Exactness of Z-two-face slab contraction and shared H2 exterior."""
import json

import numpy as np
import pytest

from betti_curves_py import persistence, slab_persistence, load_hierarchical_pairs


@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('shape,depth', [((4, 5, 6), 1), ((8, 10, 9), 2), ((11, 8, 9), 4)])
def test_two_face_exact_against_oracle(tmp_path, dim, conn, dtype, shape, depth):
    rng = np.random.default_rng(3000 + sum(shape) + dim * 9 + conn)
    volume = rng.integers(0, 13, size=shape).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        volume = (volume - 6) / 7
    oracle = persistence(volume, dim, conn, backend='numba-reference')
    result = slab_persistence(volume, dim, tmp_path / 'two-face', foreground_connectivity=conn,
                              slab_depth=depth, tile_edge=4, write_curve=False,
                              intra_slab='two-face')
    got = sorted(map(tuple, load_hierarchical_pairs(result).tolist()))
    expected = sorted(map(tuple, zip(oracle.births[:oracle.finite_count],
                                    oracle.deaths[:oracle.finite_count])))
    assert got == expected
    if dim == 0:
        assert result.essential_birth == oracle.births[-1]
    else:
        assert result.essential_birth is None
    assert json.loads(result.metadata_path.read_text())['intra_slab'] == 'two-face'


@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dim', [0, 2])
def test_two_face_random_continuous_f32(tmp_path, conn, dim):
    rng = np.random.default_rng(456 + conn + dim)
    volume = rng.normal(size=(16, 13, 17)).astype(np.float32)
    oracle = persistence(volume, dim, conn, backend='numba-reference')
    got = slab_persistence(volume, dim, tmp_path, foreground_connectivity=conn,
                           slab_depth=3, tile_edge=5, write_curve=False,
                           intra_slab='two-face')
    assert sorted(map(tuple, load_hierarchical_pairs(got).tolist())) == sorted(
        map(tuple, zip(oracle.births[:oracle.finite_count], oracle.deaths[:oracle.finite_count])))


def test_two_face_with_memmap(tmp_path):
    rng = np.random.default_rng(11)
    volume = rng.integers(0, 200, (13, 18, 21), dtype=np.uint16)
    path = tmp_path / 'volume.npy'
    np.save(path, volume)
    mmap = np.load(path, mmap_mode='r')
    for dim in (0, 2):
        reference = slab_persistence(mmap, dim, tmp_path/f'old{dim}',
                                      slab_depth=4, tile_edge=6, write_curve=False)
        new = slab_persistence(mmap, dim, tmp_path/f'new{dim}',
                              slab_depth=4, tile_edge=6, write_curve=False,
                              intra_slab='two-face')
        assert sorted(map(tuple, load_hierarchical_pairs(reference).tolist())) == sorted(
            map(tuple, load_hierarchical_pairs(new).tolist()))


@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('shape,depth', [((4, 5, 6), 1), ((8, 10, 9), 2), ((11, 8, 9), 4)])
def test_boundary_aware_exact_against_oracle(tmp_path, dim, conn, dtype, shape, depth):
    rng = np.random.default_rng(8000 + sum(shape) + dim * 9 + conn)
    volume = rng.integers(0, 15, size=shape).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        volume = (volume - 6) / 7
    oracle = persistence(volume, dim, conn, backend='numba-reference')
    result = slab_persistence(volume, dim, tmp_path / 'boundary-aware', foreground_connectivity=conn,
                              slab_depth=depth, tile_edge=4, write_curve=False,
                              intra_slab='boundary-aware')
    got = sorted(map(tuple, load_hierarchical_pairs(result).tolist()))
    expected = sorted(map(tuple, zip(oracle.births[:oracle.finite_count],
                                    oracle.deaths[:oracle.finite_count])))
    assert got == expected
    assert result.essential_birth == (oracle.births[-1] if dim == 0 else None)


@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
def test_boundary_forest_exact_against_oracle(tmp_path, dim, conn, dtype):
    rng = np.random.default_rng(9800 + dim * 17 + conn)
    vol = rng.integers(0, 23, size=(13, 15, 17)).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        vol = (vol - 13) / 9
    ref = persistence(vol, dim, conn, backend='numba-reference')
    ans = slab_persistence(vol, dim, tmp_path, foreground_connectivity=conn,
                           slab_depth=4, tile_edge=5, write_curve=False,
                           intra_slab='boundary-forest')
    assert sorted(map(tuple, load_hierarchical_pairs(ans).tolist())) == sorted(
        zip(ref.births[:ref.finite_count], ref.deaths[:ref.finite_count]))
    assert ans.essential_birth == (ref.births[-1] if dim == 0 else None)


def test_cli_boundary_aware(tmp_path):
    from betti_curves_py.cli import main
    vol = np.ones((6, 7, 8), dtype=np.uint16)
    vol[2:5, 2:6, 2:6] = 5
    vol[3, 4, 4] = 1
    inp = tmp_path / 'volume.npy'
    np.save(inp, vol)
    assert main([str(inp), '--backend', 'slab', '--intra-slab', 'boundary-aware',
                 '--mode', 'both', '--slab-depth', '2', '--tile-edge', '4',
                 '--output-dir', str(tmp_path/'results')]) == 0
    for key in ('h0', 'h2'):
        with (tmp_path/'results'/f'{key}_metadata.json').open() as file:
            assert json.load(file)['intra_slab'] == 'boundary-aware'


def test_invalid_intra_slab(tmp_path):
    vol = np.ones((4, 5, 6), np.uint8)
    with pytest.raises(ValueError):
        slab_persistence(vol, 0, tmp_path, intra_slab='bogus')
    with pytest.raises(ValueError):
        slab_persistence(vol, 0, tmp_path, intra_slab='two-face', merge_strategy='legacy')
