from __future__ import annotations

import csv
from collections import Counter

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from betti_curves_py import (
    betti_at, betti_curve, both_persistence, persistence,
    read_tiff_stack, write_betti_curve, write_branch_tree, write_persistence,
)


def intervals_counter(it):
    return Counter(it.pairs())


def expected_beta0(volume, t, fg):
    structure = ndimage.generate_binary_structure(3, 1) if fg == 6 else np.ones((3, 3, 3), bool)
    return int(ndimage.label(volume <= t, structure)[1])


def expected_beta2(volume, t, fg):
    bg = 26 if fg == 6 else 6
    structure = ndimage.generate_binary_structure(3, 1) if bg == 6 else np.ones((3, 3, 3), bool)
    labels, count = ndimage.label(volume > t, structure)
    outside = np.unique(np.concatenate([
        labels[0, :, :].ravel(), labels[-1, :, :].ravel(),
        labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ]))
    return int(count - np.count_nonzero(outside))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('fg', [6, 26])
def test_shell_exact_barcode(dtype, fg):
    volume = np.ones((3, 3, 3), dtype=dtype)
    volume[1, 1, 1] = 4
    h0, h2 = both_persistence(volume, fg)
    assert intervals_counter(h0) == Counter({(1, None): 1})
    assert intervals_counter(h2) == Counter({(1, 4): 1})
    assert betti_curve(h0) == [(1, 1)]
    assert betti_curve(h2) == [(1, 1), (4, 0)]


def test_single_voxel_and_constant_volume():
    for dtype in [np.uint8, np.float64]:
        a = np.full((1, 1, 1), 3, dtype=dtype)
        h0, h2 = both_persistence(a)
        assert h0.pairs() == [(3, None)]
        assert h2.pairs() == []


def test_signed_zero_and_float_keys():
    a = np.array([[[-0.0, 0.0, 2.5]]], dtype=np.float32)
    assert persistence(a, 0).pairs() == [(0.0, None)]
    assert persistence(a, 2).pairs() == []


@pytest.mark.parametrize('fg', [6, 26])
@pytest.mark.parametrize('backend', ['python', 'numba'])
def test_random_matches_independent_component_counts(fg, backend):
    pytest.importorskip('numba') if backend == 'numba' else None
    rng = np.random.default_rng(123)
    for _ in range(12):
        v = rng.integers(0, 6, size=(3, 4, 4), dtype=np.uint16)
        h0, h2 = both_persistence(v, fg, backend=backend)
        thresholds = np.arange(-1, 7, dtype=np.int64)
        assert betti_at(betti_curve(h0), thresholds).tolist() == [
            expected_beta0(v, t, fg) for t in thresholds
        ]
        assert betti_at(betti_curve(h2), thresholds).tolist() == [
            expected_beta2(v, t, fg) for t in thresholds
        ]
        assert all(b < d for b, d in h0.pairs() if d is not None)
        assert all(b < d for b, d in h2.pairs())


@pytest.mark.parametrize('fg', [6, 26])
def test_exhaustive_binary_2_cube(fg):
    for mask in range(256):
        v = np.array([(mask >> i) & 1 for i in range(8)], dtype=np.uint8).reshape(2, 2, 2)
        h0, h2 = both_persistence(v, fg)
        for t in (-1, 0, 1):
            assert betti_at(betti_curve(h0), np.array([t]))[0] == expected_beta0(v, t, fg)
            assert betti_at(betti_curve(h2), np.array([t]))[0] == expected_beta2(v, t, fg)
        assert h2.pairs() == []


def test_float64_scalar_distinctions():
    a = np.array([[[0.0, 1.0, np.nextafter(0.0, 1.0)]]], dtype=np.float64)
    # Distinct nearby scalar values must not be quantized together.
    h0 = persistence(a, 0)
    assert (float(np.nextafter(0.0, 1.0)), 1.0) in h0.pairs()


@pytest.mark.parametrize('bad', [np.array([[[np.nan]]], np.float64),
                                 np.array([[[np.inf]]], np.float32)])
def test_reject_nonfinite(bad):
    with pytest.raises(ValueError, match='finite'):
        persistence(bad, 0)


def test_reject_invalid_inputs():
    with pytest.raises(TypeError):
        persistence(np.array([[[1]]], dtype=np.int64), 0)
    with pytest.raises(ValueError):
        persistence(np.empty((0, 3, 3), dtype=np.uint8), 0)
    with pytest.raises(ValueError):
        persistence(np.ones((3, 3, 3), dtype=np.uint8), 1)
    with pytest.raises(ValueError):
        persistence(np.ones((3, 3, 3), dtype=np.uint8), 0, 18)


def test_stack_natural_order_and_mmap(tmp_path):
    stack = tmp_path / 'stack'
    stack.mkdir()
    for i in (10, 2, 1):
        tifffile.imwrite(stack / f'slice_{i}.tif', np.full((3, 2), i, dtype=np.uint16))
    v = read_tiff_stack(stack, mmap_path=tmp_path / 'volume.bin')
    assert isinstance(v, np.memmap)
    assert v.shape == (3, 3, 2)
    assert v[:, 0, 0].tolist() == [1, 2, 10]
    assert (tmp_path / 'volume.bin').exists()


def test_tiff_bad_types_and_pages(tmp_path):
    root = tmp_path / 'bad'
    root.mkdir()
    tifffile.imwrite(root / 'slice0.tif', np.ones((4, 4), dtype=np.uint16))
    tifffile.imwrite(root / 'slice1.tif', np.ones((5, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match='inconsistent'):
        read_tiff_stack(root)


def test_csv_and_trees(tmp_path):
    v = np.array([[[0, 4, 1, 4, 2]]], dtype=np.uint16)
    h0 = persistence(v, 0, branch_tree=True)
    write_persistence(tmp_path / 'h0.csv', h0)
    write_betti_curve(tmp_path / 'curve.csv', h0)
    nodes, edges = write_branch_tree(tmp_path / 'tree', h0)
    with (tmp_path / 'h0.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert {'birth': '0', 'death': 'inf'} in rows
    assert {'birth': '1', 'death': '4'} in rows
    assert {'birth': '2', 'death': '4'} in rows
    with nodes.open() as file:
        tree = list(csv.DictReader(file))
    assert len(tree) == len(rows)
    assert sum(row['parent'] == '' for row in tree) == 1
    with edges.open() as file:
        assert len(list(csv.DictReader(file))) == len(tree) - 1

    shell = np.ones((3, 3, 3), dtype=np.uint8)
    shell[1, 1, 1] = 5
    h2 = persistence(shell, 2, branch_tree=True)
    nodes, _ = write_branch_tree(tmp_path / 'h2tree', h2)
    with nodes.open() as file:
        tree = list(csv.DictReader(file))
    assert tree[0]['birth_value'] == '-inf' and tree[0]['death_value'] == 'inf'
    assert tree[1]['birth_value'] == '1' and tree[1]['death_value'] == '5'
    assert tree[1]['parent'] == '0'


def test_empty_h2_branch_tree(tmp_path):
    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    h2 = persistence(volume, 2, branch_tree=True)
    nodes, edges = write_branch_tree(tmp_path / 'emptyh2', h2)
    with nodes.open() as file:
        assert list(csv.DictReader(file)) == [
            {'node': '0', 'parent': '', 'birth_value': '-inf', 'death_value': 'inf'}
        ]
    with edges.open() as file:
        assert list(csv.DictReader(file)) == []


@pytest.mark.parametrize('fg', [6, 26])
@pytest.mark.parametrize('dimension', [0, 2])
def test_random_tree_is_acyclic_and_connected(fg, dimension, tmp_path):
    rng = np.random.default_rng(858)
    for run in range(25):
        volume = rng.integers(0, 9, size=(4, 4, 5), dtype=np.uint16)
        intervals = persistence(volume, dimension, fg, branch_tree=True)
        nodes, _ = write_branch_tree(tmp_path / f'tree_{run}', intervals)
        with nodes.open() as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == len(intervals) + int(dimension == 2)
        assert sum(not row['parent'] for row in rows) == 1
        parents = {int(row['node']): int(row['parent'])
                   for row in rows if row['parent']}
        for node in parents:
            current = node
            visited = set()
            while current in parents:
                assert current not in visited
                visited.add(current)
                current = parents[current]
        assert len(parents) == len(rows) - 1


def test_cli_from_generated_tiff_stack(tmp_path):
    from betti_curves_py.cli import main

    root = tmp_path / 'inputs'
    root.mkdir()
    v = np.zeros((3, 3, 3), dtype=np.uint8)
    v[1, 1, 1] = 1
    for k in range(3):
        tifffile.imwrite(root / f'img_{k:03}.tif', v[k])
    result = tmp_path / 'outputs'
    assert main([str(root), '--mode', 'tree-both', '--output-dir', str(result)]) == 0
    with (result / 'h2_persistence.csv').open() as handle:
        assert list(csv.DictReader(handle)) == [{'birth': '0', 'death': '1'}]
    for path in ('h0_persistence.csv', 'h2_betti_curve.csv',
                 'h0_branch_tree_nodes.csv', 'h2_branch_tree_nodes.csv'):
        assert (result / path).exists()
