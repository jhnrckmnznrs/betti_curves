"""Cross-validate rolling Z-frontier persistence against the global oracle."""
import csv
import json

import numpy as np
import pytest

from betti_curves_py import persistence, betti_curve, load_hierarchical_pairs, slab_persistence
from betti_curves_py.cli import main


@pytest.mark.parametrize('dim', (0, 2))
@pytest.mark.parametrize('connectivity', (6, 26))
@pytest.mark.parametrize('dtype', (np.uint8, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize('shape', ((1, 2, 3), (3, 5, 7), (8, 9, 10)))
@pytest.mark.parametrize('slab_depth', (1, 3))
def test_slab_matches_global(tmp_path, dim, connectivity, dtype, shape, slab_depth):
    rng = np.random.default_rng(sum(shape) * 123 + connectivity + dim)
    values = rng.integers(0, 11, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        values = values / 5 - 0.8
    ref = persistence(values, dim, connectivity, backend='numba-reference')
    result = slab_persistence(values, dim, tmp_path,
                              foreground_connectivity=connectivity,
                              slab_depth=slab_depth, tile_edge=3)
    got = sorted(map(tuple, load_hierarchical_pairs(result).tolist()))
    expected = sorted(zip(ref.births[:ref.finite_count], ref.deaths[:ref.finite_count]))
    assert got == expected
    assert result.pair_count == ref.finite_count
    if dim == 0:
        assert result.essential_birth == ref.births[-1]
    else:
        assert result.essential_birth is None
    with result.curve_path.open() as file:
        rows = list(csv.reader(file))
    assert [(float(a), int(b)) for a, b in rows[1:]] == betti_curve(ref)
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata['slabs'] == (shape[0] + slab_depth - 1) // slab_depth


@pytest.mark.parametrize('dim', (0, 2))
@pytest.mark.parametrize('conn', (6, 26))
def test_random_float_continuous_and_memmap(tmp_path, dim, conn):
    rng = np.random.default_rng(100 + conn + dim)
    values = rng.uniform(-7, 11, (12, 11, 16)).astype('float32')
    file = tmp_path / 'volume.npy'
    np.save(file, values)
    src = np.load(file, mmap_mode='r')
    ref = persistence(values, dim, conn, backend='numba')
    got = slab_persistence(src, dim, tmp_path / 'results', foreground_connectivity=conn,
                           slab_depth=4, tile_edge=4, write_curve=False)
    assert sorted(map(tuple, load_hierarchical_pairs(got).tolist())) == sorted(
        zip(ref.births[:ref.finite_count], ref.deaths[:ref.finite_count]))
    assert got.curve_path is None


def test_cli_slab(tmp_path):
    a = np.ones((7, 7, 7), np.uint16)
    a[2:5,2:5,2:5] = 4
    a[3,3,3] = 1
    file = tmp_path / 'a.npy'
    np.save(file, a)
    assert main([str(file), '--backend', 'slab', '--mode', 'both',
                 '--slab-depth', '2', '--tile-edge', '3',
                 '--output-dir', str(tmp_path / 'outputs')]) == 0
    assert (tmp_path / 'outputs' / 'h0_pairs.bin').exists()
    assert (tmp_path / 'outputs' / 'h2_pairs.bin').exists()


def test_slab_invalid_parameters(tmp_path):
    a = np.ones((3, 4, 5), np.uint8)
    for kwargs in ({'slab_depth':0}, {'tile_edge':1}, {'merge_strategy':'other'},
                   {'foreground_connectivity':12}):
        with pytest.raises(ValueError):
            slab_persistence(a, 0, tmp_path, **kwargs)

@pytest.mark.parametrize('connectivity', (6, 26))
def test_filtered_seam_spanning_forest_edge_bound(connectivity):
    from betti_curves_py.slab import _seam_spanning_forest
    from betti_curves_py.optimized import stable_voxel_order
    height, width = 8, 9
    n = height * width
    values = np.ones(2 * n, dtype=np.uint16)
    u, v, levels = _seam_spanning_forest(
        values, stable_voxel_order(values, True), width, connectivity, True, 0, n)
    assert len(u) == (n if connectivity == 6 else 2*n-1)
    assert len(u) == len(v) == len(levels)
    assert (levels == 1).all()
