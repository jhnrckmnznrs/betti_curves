"""Check multilevel contraction against full exact persistence and independent IO."""
import csv
import json

import numpy as np
import pytest

from betti_curves_py import (
    betti_curve, hierarchical_persistence, load_hierarchical_pairs, persistence,
)
from betti_curves_py.cli import main


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 3, 5), (2, 2, 2), (3, 5, 7), (6, 8, 9), (9, 8, 6)])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('edge', [2, 4])
def test_hierarchical_matches_full(tmp_path, dtype, shape, conn, dim, edge):
    rng = np.random.default_rng(919 + sum(shape) + int(conn) + int(dim))
    scalar = rng.integers(0, 7, size=shape).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        scalar = (scalar - 3) / 2
    result = hierarchical_persistence(scalar, dim, tmp_path, foreground_connectivity=conn, tile_edge=edge)
    reference = persistence(scalar, dim, conn, backend='numba-reference')
    expected = sorted(zip(reference.births[:reference.finite_count].tolist(),
                          reference.deaths[:reference.finite_count].tolist()))
    got = sorted(map(tuple, load_hierarchical_pairs(result).tolist()))
    assert got == expected
    if dim == 0:
        assert result.essential_birth == reference.births[-1]
    else:
        assert result.essential_birth is None
    assert result.pair_count == reference.finite_count
    assert result.metadata_path.exists()
    assert result.curve_path is not None
    curve = list(csv.reader(result.curve_path.open()))
    expected_curve = betti_curve(reference)
    assert [(float(x), int(y)) for x, y in curve[1:]] == expected_curve


def test_hierarchical_memmap_large_random(tmp_path):
    rng = np.random.default_rng(805)
    array = rng.integers(0, 256, size=(24, 25, 26), dtype=np.uint8)
    source = tmp_path / 'input.npy'
    np.save(source, array)
    view = np.load(source, mmap_mode='r')
    reference = persistence(array, 2, 6, backend='numba')
    result = hierarchical_persistence(view, 2, tmp_path / 'output', tile_edge=6)
    assert sorted(map(tuple, load_hierarchical_pairs(result).tolist())) == sorted(
        zip(reference.births.tolist(), reference.deaths.tolist()))
    assert result.peak_summary_nodes < array.size
    assert result.tiles > 1
    meta = json.loads(result.metadata_path.read_text())
    assert meta['finite_pairs'] == reference.finite_count


def test_float_external_curve_many_distinct_levels(tmp_path):
    rng = np.random.default_rng(100)
    a = rng.uniform(-2, 2, size=(8, 9, 10)).astype(np.float64)
    for dim in (0, 2):
        reference = persistence(a, dim, 26, backend='numba')
        result = hierarchical_persistence(a, dim, tmp_path / str(dim), tile_edge=3, foreground_connectivity=26)
        with result.curve_path.open() as f:
            rows = list(csv.reader(f))
        assert [(float(t), int(beta)) for t, beta in rows[1:]] == betti_curve(reference)


def test_no_curve_and_input_validation(tmp_path):
    arr = np.zeros((4, 3, 2), dtype=np.uint16)
    result = hierarchical_persistence(arr, 0, tmp_path, write_curve=False)
    assert result.curve_path is None
    with pytest.raises(ValueError, match='NaN/Inf'):
        invalid = np.ones((5, 5, 5), dtype=np.float32)
        invalid[3, 4, 4] = np.nan
        hierarchical_persistence(invalid, 0, tmp_path, tile_edge=2)
    with pytest.raises(ValueError, match='tile_edge'):
        hierarchical_persistence(arr, 0, tmp_path, tile_edge=1)
    with pytest.raises(TypeError):
        hierarchical_persistence(arr.astype(np.int16), 0, tmp_path)


def test_cli_hierarchical_numpy(tmp_path):
    arr = np.ones((5, 5, 5), dtype=np.uint16)
    arr[2, 2, 2] = 4
    source = tmp_path / 'volume.npy'
    output = tmp_path / 'results'
    np.save(source, arr)
    assert main([str(source), '--mode', 'both', '--backend', 'hierarchical',
                 '--tile-edge', '2', '--output-dir', str(output)]) == 0
    assert (output / 'h0_pairs.bin').exists()
    assert (output / 'h2_pairs.bin').exists()
    assert list(csv.reader((output / 'h2_betti_curve.csv').open())) == [
        ['threshold', 'betti2'], ['1', '1'], ['4', '0']
    ]


def test_csv_export_hierarchical(tmp_path):
    from betti_curves_py import export_hierarchical_pairs_csv
    data = np.ones((3, 3, 3), dtype=np.uint8)
    data[1, 1, 1] = 4
    for dim in (0, 2):
        got = hierarchical_persistence(data, dim, tmp_path / str(dim), tile_edge=2)
        target = export_hierarchical_pairs_csv(got, tmp_path / f'{dim}.csv', chunk_pairs=1)
        assert list(csv.reader(target.open())) == (
            [['birth', 'death'], ['1', 'inf']] if dim == 0
            else [['birth', 'death'], ['1', '4']]
        )


def test_float_curve_external_multipass(tmp_path):
    from betti_curves_py.hierarchical import _write_float_curve_external
    from betti_curves_py import betti_curve, Intervals
    births = np.array([-1.0, -0.5, -0.1, 0.3, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0],
                      dtype=np.float64)
    deaths = births + 2.5
    pairs = np.column_stack((births, deaths))
    binary = tmp_path / 'pairs.bin'
    pairs.tofile(binary)
    destination = tmp_path / 'curve.csv'
    _write_float_curve_external(binary, destination, dtype=np.float64,
                                pair_count=len(births), dimension=0,
                                essential_birth=-2.0, chunk_pairs=2,
                                merge_fan_in=3)
    expected = betti_curve(Intervals(
        np.concatenate((births, [-2.0])),
        np.concatenate((deaths, [0.])), dimension=0, essential=True))
    rows = list(csv.reader(destination.open()))
    assert [(float(t), int(beta)) for t, beta in rows[1:]] == expected


def test_hierarchical_cli_tiff_stack(tmp_path):
    import tifffile
    folder = tmp_path / 'tiff'
    folder.mkdir()
    data = np.ones((3, 3, 3), np.uint16)
    data[1, 1, 1] = 5
    for z in range(len(data)):
        tifffile.imwrite(folder / f'slice_{z}.tif', data[z])
    result = tmp_path / 'output'
    assert main([str(folder), '--backend', 'hierarchical', '--mode', 'h2',
                 '--tile-edge', '2', '--output-dir', str(result), '--pairs-csv']) == 0
    with (result / 'h2_persistence.csv').open() as handle:
        assert list(csv.reader(handle)) == [['birth', 'death'], ['1', '5']]
