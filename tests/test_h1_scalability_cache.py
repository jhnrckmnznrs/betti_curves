"""Disk-backed exact H1 pair lookup and memory-bounded catalog tests."""
import json
import os
import numpy as np
import pytest
from betti_curves_py.h1_morse import h1_persistence
from betti_curves_py._morse.native_backend import CompactMorse


def two_tunnels():
    volume = np.full((12, 12, 12), 5, dtype=np.uint8)
    volume[1:5, 1:5, 1:5] = 0
    volume[7:11, 7:11, 7:11] = 0
    volume[1:5, 2, 2] = 5
    volume[7:11, 8, 8] = 5
    return volume


def test_saved_catalog_reuse_and_pair_lookup(tmp_path, monkeypatch):
    result = h1_persistence(two_tunnels(), output_dir=tmp_path)
    morse = CompactMorse.open(tmp_path)
    before = (tmp_path / 'h1_pairs.bin').stat().st_mtime_ns
    # Barcode computation alone does not build a per-pair index.
    assert not (tmp_path / 'h1_creators.npy').exists()
    morse.h1_pair(result.pair_ids[0])
    index = np.load(tmp_path / 'h1_creators.npy', mmap_mode='r')
    assert len(index) == 2 and np.all(index['creator'][:-1] < index['creator'][1:])
    assert sorted(result.pairs()) == [(0, 5), (0, 5)]
    assert set(result.pair_ids) == set(morse.h1_pair_at(i)[0] for i in range(2))
    # On an existing complete catalog, native compilation must not be needed
    # merely to enumerate or look up individual pair records.
    import betti_curves_py._morse.native_backend as nb
    def no_native(*args, **kwargs):
        raise AssertionError('Full native cohomology was unexpectedly rerun')
    monkeypatch.setattr(nb, 'build_native', no_native)
    for pair_id in result.pair_ids:
        rec = morse.h1_pair(pair_id)
        assert rec[0] == pair_id and rec[1:3] == (0, 5)
    assert set(x[0] for x in morse.h1_pairs()) == set(result.pair_ids)
    assert (tmp_path / 'h1_pairs.bin').stat().st_mtime_ns == before


def test_memory_bounded_ids_and_reconstruction(tmp_path):
    result = h1_persistence(two_tunnels(), output_dir=tmp_path, store_pair_ids=False)
    assert result.compact_pair_ids and result.pair_ids == ()
    assert len(result) == 2
    ids = tuple(result.iter_pair_ids())
    assert ids == tuple(result.pair_id_at(i) for i in range(len(result)))
    assert len(set(ids)) == 2
    assert result.pairs() == [(0, 5), (0, 5)]
    for pair in ids:
        cert = result.reconstruct(pair)
        assert cert['verified'] and cert['pair_id'] == pair
    with pytest.raises(IndexError):
        result.pair_id_at(2)


def test_invalid_pair_ids_are_rejected(tmp_path):
    result = h1_persistence(two_tunnels(), output_dir=tmp_path, store_pair_ids=False)
    morse = CompactMorse.open(tmp_path)
    for bad in (None, 'garbage', '-1:4', '123:', '1:2:3', '4294967296:1', '1:4294967296'):
        with pytest.raises(ValueError):
            morse.h1_pair(bad)
    pid = result.pair_id_at(0)
    start, end = map(int, pid.split(':'))
    with pytest.raises(ValueError):
        morse.h1_pair(f'{start}:{end+1}')


def test_interrupted_cache_metadata_cannot_be_reused(tmp_path):
    result = h1_persistence(two_tunnels(), output_dir=tmp_path)
    path = tmp_path / 'h1_pairs_metadata.json'
    info = json.loads(path.read_text())
    assert info['count'] == 2
    morse = CompactMorse.open(tmp_path)
    assert len(morse._cached_h1_records()) == 2
    info['count'] += 1
    path.write_text(json.dumps(info))
    assert morse._cached_h1_records() is None
    assert sorted((p[1], p[2]) for p in morse.h1_pairs()) == sorted(result.pairs())
    assert json.loads(path.read_text())['count'] == 2


def test_changed_complex_forces_new_cohomology(tmp_path):
    h1_persistence(two_tunnels(), output_dir=tmp_path)
    morse = CompactMorse.open(tmp_path)
    p = tmp_path / 'critical_index.bin'
    old = p.stat()
    os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns + 10_000_000))
    assert morse._cached_h1_records() is None
    assert len(morse.h1_pairs()) == 2


def test_empty_pair_catalog_is_cached(tmp_path):
    h1_persistence(np.zeros((3,3,3), dtype=np.uint8), output_dir=tmp_path)
    morse = CompactMorse.open(tmp_path)
    assert len(morse.h1_records()) == 0
    assert len(morse.h1_pairs()) == 0
    with pytest.raises(ValueError):
        morse.h1_pair('0:0')
