"""Disk-backed worker reconciliation and compact boundary ID regression tests."""
import csv
import json

import numpy as np
import pytest

from betti_curves_py import hierarchical_persistence, load_hierarchical_pairs, persistence
from betti_curves_py.cli import main
from betti_curves_py.hierarchical import _cross_ids, _merge_vertex_sets


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('conn', [6, 26])
@pytest.mark.parametrize('dim', [0, 2])
def test_parallel_exact_intervals_and_curve(tmp_path, dtype, conn, dim):
    rng = np.random.default_rng(301 + conn + dim)
    a = rng.integers(0, 11, size=(8, 9, 10), dtype=np.uint16).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        a = (a - 5) / 4
    np.save(tmp_path / 'volume.npy', a)
    volume = np.load(tmp_path / 'volume.npy', mmap_mode='r')
    parallel = hierarchical_persistence(
        volume, dim, tmp_path / 'parallel', foreground_connectivity=conn,
        tile_edge=3, parallel_workers=2, compact_ids=True)
    serial = hierarchical_persistence(
        volume, dim, tmp_path / 'serial', foreground_connectivity=conn,
        tile_edge=3, parallel_workers=1, compact_ids=False)
    reference = persistence(a, dim, conn, backend='numba-reference')
    expected = sorted(zip(reference.births[:reference.finite_count].tolist(),
                          reference.deaths[:reference.finite_count].tolist()))
    assert sorted(map(tuple, load_hierarchical_pairs(parallel).tolist())) == expected
    assert sorted(map(tuple, load_hierarchical_pairs(serial).tolist())) == expected
    assert parallel.essential_birth == serial.essential_birth
    assert parallel.pair_count == serial.pair_count
    assert list(csv.reader(parallel.curve_path.open())) == list(csv.reader(serial.curve_path.open()))
    assert parallel.tiles == serial.tiles
    metadata = json.loads(parallel.metadata_path.read_text())
    assert metadata['global_id_dtype'] == 'uint32'
    assert metadata['parallel_workers'] == 2
    assert not metadata['parallel_input_spooled']
    assert not list((tmp_path / 'parallel').glob('betti_workers_*'))


def test_parallel_spooled_source_and_no_curve(tmp_path):
    volume = np.zeros((7, 8, 6), np.uint16)
    volume[2:6, 2:5, 1:4] = 5
    volume[3:5, 3:4, 2:3] = 10
    result = hierarchical_persistence(volume, 2, tmp_path / 'parallel',
                                       parallel_workers=3, tile_edge=3, write_curve=False)
    serial = hierarchical_persistence(volume, 2, tmp_path / 'serial',
                                      tile_edge=3, write_curve=False)
    assert sorted(map(tuple, load_hierarchical_pairs(result).tolist())) == sorted(
        map(tuple, load_hierarchical_pairs(serial).tolist()))
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata['parallel_input_spooled'] is False
    assert result.curve_path is None
    proc = hierarchical_persistence(volume, 2, tmp_path / 'process',
                                    parallel_workers=2, parallel_backend='processes',
                                    tile_edge=3, write_curve=False)
    assert sorted(map(tuple, load_hierarchical_pairs(proc).tolist())) == sorted(
        map(tuple, load_hierarchical_pairs(serial).tolist()))
    assert json.loads(proc.metadata_path.read_text())['parallel_input_spooled'] is True


def test_parallel_invalid_float_propagates(tmp_path):
    array = np.ones((5, 6, 7), np.float32)
    array[2, 3, 5] = np.inf
    with pytest.raises(ValueError, match='NaN/Inf'):
        hierarchical_persistence(array, 0, tmp_path / 'bad', tile_edge=2, parallel_workers=2)


def test_worker_count_validation(tmp_path):
    with pytest.raises(ValueError, match='parallel_workers'):
        hierarchical_persistence(np.ones((2, 2, 2), np.uint8), 0,
                                 tmp_path / 'bad', parallel_workers=0)


def test_parallel_cli_npy(tmp_path):
    source = tmp_path / 'volume.npy'
    data = np.ones((6, 7, 8), np.uint8)
    data[2:4, 2:5, 2:6] = 4
    data[3, 3, 3] = 7
    np.save(source, data)
    output = tmp_path / 'output'
    assert main([str(source), '--backend', 'hierarchical', '--mode', 'both',
                 '--tile-edge', '3', '--workers', '2', '--output-dir', str(output)]) == 0
    assert (output / 'h0_pairs.bin').exists()
    assert (output / 'h2_pairs.bin').exists()


def test_merged_sorted_id_maps():
    left = np.array([0, 3, 8, 12], np.uint32)
    right = np.array([1, 2, 6, 9, 13], np.uint32)
    lv = np.array([2, 5, 6, 1], np.uint16)
    rv = np.array([8, 7, 4, 3, 9], np.uint16)
    ids, vals, lm, rm = _merge_vertex_sets(left, right, lv, rv)
    assert ids.tolist() == sorted(left.tolist() + right.tolist())
    np.testing.assert_array_equal(ids[lm], left)
    np.testing.assert_array_equal(ids[rm], right)
    np.testing.assert_array_equal(vals[lm], lv)
    np.testing.assert_array_equal(vals[rm], rv)


@pytest.mark.parametrize('conn', [6, 26])
def test_cross_edges_equivalent_to_direct_enumeration(conn):
    shape = (5, 6, 7)
    box = ((0, 5), (0, 6), (0, 7))
    for axis, mid in [(0, 2), (1, 3), (2, 4)]:
        a, b = _cross_ids(box, axis, mid, shape, conn, np.uint32)
        observed = set(zip(a.tolist(), b.tolist()))
        assert len(observed) == len(a)
        brute = set()
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    p = (z, y, x)
                    if p[axis] != mid - 1:
                        continue
                    for dz in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                q = (z + dz, y + dy, x + dx)
                                if any(t < 0 or t >= shape[i] for i, t in enumerate(q)):
                                    continue
                                if q[axis] != mid:
                                    continue
                                if conn == 6 and abs(dx) + abs(dy) + abs(dz) != 1:
                                    continue
                                def pack(v):
                                    return (v[0] * shape[1] + v[1]) * shape[2] + v[2]
                                brute.add((pack(p), pack(q)))
        assert observed == brute

@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('conn', [6, 26])
def test_four_threads_continuous_float_against_reference(tmp_path, dim, conn):
    rng = np.random.default_rng(564 + conn + dim)
    array = rng.uniform(-1, 2, size=(18, 19, 20)).astype(np.float32)
    np.save(tmp_path / 'input.npy', array)
    mapped = np.load(tmp_path / 'input.npy', mmap_mode='r')
    got = hierarchical_persistence(mapped, dim, tmp_path / 'result',
                                   tile_edge=6, foreground_connectivity=conn,
                                   parallel_workers=4)
    ref = persistence(array, dim, conn, backend='numba')
    assert sorted(map(tuple, load_hierarchical_pairs(got).tolist())) == sorted(
        zip(ref.births[:ref.finite_count].tolist(), ref.deaths[:ref.finite_count].tolist()))
    if dim == 0:
        assert got.essential_birth == ref.births[-1]


def test_large_unsigned_global_ids_and_limit():
    from betti_curves_py.hierarchical import _id_dtype
    assert _id_dtype((1024, 1024, 2048), True) is np.uint32
    assert _id_dtype((2048, 2048, 1024), True) is np.int64
    assert _id_dtype((2, 2, 2), False) is np.int64
    a = np.array([2**31 + 1, 2**32 - 2], np.uint32)
    b = np.array([2**31 + 2, 2**32 - 1], np.uint32)
    v = np.array([2, 3], np.uint8)
    ids, _, lmap, rmap = _merge_vertex_sets(a, b, v, v)
    np.testing.assert_array_equal(ids[lmap], a)
    np.testing.assert_array_equal(ids[rmap], b)
