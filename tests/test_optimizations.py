"""Differential, boundary, and fast-path regression tests."""
from __future__ import annotations

import csv

import numpy as np
import pytest
import tifffile

from betti_curves_py import betti_at, betti_curve, persistence, streaming_betti_at
from betti_curves_py.core import Intervals
from betti_curves_py.io import _branch_rows, write_branch_tree
from betti_curves_py.optimized import stable_voxel_order


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('size', [13, 131073])
def test_sort_stability_and_integer_counting(dtype, descending, size):
    pytest.importorskip('numba')
    rng = np.random.default_rng(1221)
    if dtype == np.uint8:
        values = rng.integers(0, 250, size=size, dtype=dtype)
    elif dtype == np.uint16:
        values = rng.integers(0, 65535, size=size, dtype=dtype)
    else:
        values = rng.integers(-80, 80, size=size).astype(dtype)
    expected = np.argsort(-values.astype(np.float64), kind='stable') if descending else np.argsort(values, kind='stable')
    for counting in (False, True):
        order = stable_voxel_order(values, descending, counting=counting)
        assert np.array_equal(order, expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('connectivity', [6, 26])
@pytest.mark.parametrize('dimension', [0, 2])
@pytest.mark.parametrize('tree', [False, True])
def test_optimized_sweep_exactly_matches_reference(dtype, connectivity, dimension, tree):
    pytest.importorskip('numba')
    rng = np.random.default_rng(317)
    # Many distinct component births, plus repeated values to stress tie rules.
    for shape in [(1, 4, 6), (3, 5, 7), (7, 5, 3)]:
        volume = rng.integers(0, 30, size=shape).astype(dtype)
        x = persistence(volume, dimension, connectivity, backend='numba', branch_tree=tree)
        y = persistence(volume, dimension, connectivity, backend='numba-reference', branch_tree=tree)
        assert np.array_equal(x.births, y.births)
        assert np.array_equal(x.deaths, y.deaths)
        if tree:
            assert np.array_equal(x.branch_ids, y.branch_ids)
            assert np.array_equal(x.parent_branch_ids, y.parent_branch_ids)


def test_large_dynamic_buffer_matches_reference():
    pytest.importorskip('numba')
    volume = np.random.default_rng(77).integers(0, 65536, size=(60, 80, 64), dtype=np.uint16)
    for dim in (0, 2):
        x = persistence(volume, dim, 6, backend='numba', branch_tree=True)
        y = persistence(volume, dim, 6, backend='numba-reference', branch_tree=True)
        assert len(x) > 4096
        for field in ('births', 'deaths', 'branch_ids', 'parent_branch_ids'):
            assert np.array_equal(getattr(x, field), getattr(y, field))


def test_float64_extreme_and_signed_zero():
    volume = np.array([[[0.0, -0.0, np.nextafter(0., 1.),
                         -np.finfo(np.float64).max, np.finfo(np.float64).max]]])
    for dim in (0, 2):
        x = persistence(volume, dim, backend='numba')
        y = persistence(volume, dim, backend='numba-reference')
        assert x.pairs() == y.pairs()


def test_dense_betti_histogram_cancellation():
    it = Intervals(births=np.array([1, 2, 2, 5], dtype=np.uint16),
                   deaths=np.array([2, 3, 5, 0], dtype=np.uint16),
                   dimension=0, essential=True)
    assert betti_curve(it) == [(1, 1), (2, 2), (3, 1)]
    # At threshold 5, one interval ends and the essential one begins;
    # no change occurs, so the threshold should be absent.
    assert betti_at(betti_curve(it), np.array([-1, 1, 2, 3, 5, 100])).tolist() == [0, 1, 2, 1, 1, 1]


def test_streaming_numba_and_python_agree(tmp_path):
    pytest.importorskip('numba')
    rng = np.random.default_rng(4044)
    for shape, dtype in [((9, 11, 12), np.uint16), ((5, 8, 7), np.float64)]:
        volume = rng.integers(0, 15, size=shape).astype(dtype)
        root = tmp_path / f'volume_{dtype.__name__}'
        root.mkdir()
        for z, image in enumerate(volume):
            tifffile.imwrite(root / f'{z:04}.tif', image)
        for dim in (0, 2):
            for connectivity in (6, 26):
                levels = [-1, 0, 3, 7, 10, 15]
                expected = betti_at(betti_curve(persistence(volume, dim, connectivity)), np.asarray(levels))
                for depth in (1, 3, 5):
                    fast = streaming_betti_at(root, levels, dimension=dim,
                        foreground_connectivity=connectivity, slab_depth=depth, backend='numba')
                    slow = streaming_betti_at(root, levels, dimension=dim,
                        foreground_connectivity=connectivity, slab_depth=depth, backend='python')
                    assert fast == slow
                    assert [v for _, v in fast] == expected.tolist()


def test_branch_tree_canonical_reference(tmp_path):
    # Independent old-style chain-walk oracle for the plateau skip cache.
    rng = np.random.default_rng(827)
    for dim in (0, 2):
        for k in range(45):
            vol = rng.integers(0, 10, size=(4, 5, 6), dtype=np.uint16)
            intervals = persistence(vol, dim, branch_tree=True)
            assert intervals.branch_ids is not None
            branches = [int(v) for v in intervals.branch_ids]
            parent = dict(zip(branches, (int(v) for v in intervals.parent_branch_ids)))
            bounds = dict(zip(branches, zip(intervals.births.tolist(), intervals.deaths.tolist())))
            if dim == 2:
                roots = [p for p in parent.values() if p not in parent]
                outside = roots[0] if roots else -2
                parent[outside] = -1
                bounds[outside] = ('-inf', 'inf')
                ordered = [outside] + sorted(branches)
            else:
                ordered = sorted(branches)
            dense = {key: i for i, key in enumerate(ordered)}
            old_rows = []
            col = 0 if dim == 2 else 1
            for ident in ordered:
                p = parent[ident]
                nsteps = 0
                while p != -1 and bounds[p][col] == bounds[ident][col]:
                    p = parent[p]
                    nsteps += 1
                    assert nsteps <= len(ordered)
                b, d = bounds[ident]
                if dim == 2 and ident == outside:
                    b, d = '-inf', 'inf'
                elif dim == 0 and ident == int(intervals.branch_ids[-1]):
                    d = 'inf'
                old_rows.append((dense[ident], '' if p == -1 else dense[p], b, d))
            assert _branch_rows(intervals, canonicalize=True) == old_rows
            assert len(write_branch_tree(tmp_path / f'tree_{dim}_{k}', intervals)) == 2
