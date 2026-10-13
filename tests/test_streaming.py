from __future__ import annotations

import numpy as np
import pytest
import tifffile

from betti_curves_py import betti_at, betti_curve, persistence, streaming_betti_at


@pytest.mark.parametrize('dtype', [np.uint8, np.float32])
@pytest.mark.parametrize('foreground', [6, 26])
@pytest.mark.parametrize('dimension', [0, 2])
@pytest.mark.parametrize('slab_depth', [1, 2, 4, 5])
def test_streaming_exact_counts_vs_persistence(tmp_path, dtype, foreground, dimension, slab_depth):
    rng = np.random.default_rng(3013)
    for case in range(4):
        volume = rng.integers(0, 8, size=(5, 4, 5)).astype(dtype)
        root = tmp_path / f'stack_{case}'
        root.mkdir()
        for k, image in enumerate(volume):
            tifffile.imwrite(root / f'slice_{k:04d}.tiff', image)
        thresholds = [-2, 0, 1, 3, 4, 7, 8]
        result = streaming_betti_at(root, thresholds, dimension=dimension,
                                    foreground_connectivity=foreground,
                                    slab_depth=slab_depth)
        curve = betti_curve(persistence(volume, dimension, foreground))
        assert [beta for _, beta in result] == betti_at(curve, np.array(thresholds)).tolist()


def test_streaming_single_voxel_and_empty_background(tmp_path):
    root = tmp_path / 'stack'
    root.mkdir()
    tifffile.imwrite(root / 'slice_0.tif', np.array([[1]], dtype=np.uint16))
    assert streaming_betti_at(root, [0, 1, 3], dimension=0) == [(0, 0), (1, 1), (3, 1)]
    assert streaming_betti_at(root, [0, 1, 3], dimension=2) == [(0, 0), (1, 0), (3, 0)]


def test_streaming_errors(tmp_path):
    with pytest.raises(ValueError, match='slab_depth'):
        streaming_betti_at(tmp_path, [1], slab_depth=0)
    with pytest.raises(ValueError, match='at least one'):
        streaming_betti_at(tmp_path, [], dimension=2)


def test_cli_streaming_mode(tmp_path):
    from csv import DictReader
    from betti_curves_py.cli import main

    root = tmp_path / 'stack'
    root.mkdir()
    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    volume[1, 1, 1] = 4
    for z in range(3):
        tifffile.imwrite(root / f'slice_{z:04}.tif', volume[z])
    out = tmp_path / 'counts'
    assert main([str(root), '--mode', 'threshold-both', '--thresholds=-1,0,1,4',
                 '--slab-depth', '1', '--output-dir', str(out)]) == 0
    with (out / 'h0_threshold_counts.csv').open() as f:
        assert list(DictReader(f)) == [
            {'threshold': '-1', 'betti0': '0'},
            {'threshold': '0', 'betti0': '1'},
            {'threshold': '1', 'betti0': '1'},
            {'threshold': '4', 'betti0': '1'},
        ]
    with (out / 'h2_threshold_counts.csv').open() as f:
        assert list(DictReader(f)) == [
            {'threshold': '-1', 'betti2': '0'},
            {'threshold': '0', 'betti2': '1'},
            {'threshold': '1', 'betti2': '1'},
            {'threshold': '4', 'betti2': '0'},
        ]
