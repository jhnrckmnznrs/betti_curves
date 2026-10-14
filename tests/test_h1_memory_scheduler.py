"""Bounded native Morse work and independent closed-voxel validation."""
import json
import numpy as np
import pytest

from betti_curves_py.h1_morse import h1_persistence
from betti_curves_py._morse.native_backend import CompactMorse


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_indexed_matches_legacy_and_exact_small(seed, dtype, tmp_path):
    a = np.random.default_rng(31 + seed).integers(0, 7, (4, 4, 4), dtype=dtype)
    reference = h1_persistence(a, output_dir=tmp_path / 'original', indexed_scheduler=False)
    for cap in (1, 50, 100000):
        result = h1_persistence(
            a, output_dir=tmp_path / f'indexed-{cap}',
            indexed_scheduler=True, max_heap_entries=cap,
        )
        assert sorted(result.pairs()) == sorted(reference.pairs())
        assert np.array_equal(result.at(np.arange(7)), reference.at(np.arange(7)))
        if len(result) and cap == 100000:
            certificate = result.reconstruct(result.pair_ids[0])
            assert certificate['verified']


def test_bounded_scheduler_may_leave_more_cells_but_not_change_barcode(tmp_path):
    a = np.random.default_rng(19).integers(0, 8, (16, 16, 16), dtype=np.uint8)
    tiny = h1_persistence(a, output_dir=tmp_path/'tiny', indexed_scheduler=True,
                          max_heap_entries=1)
    full = h1_persistence(a, output_dir=tmp_path/'full', indexed_scheduler=True)
    assert tiny.statistics['hit_budget'] == 1
    # Greedy cancellation order changes with budget; smaller heaps need not
    # leave more critical cells, even though they may in other examples.
    assert 0 < tiny.statistics['remaining'] <= tiny.statistics['precollapse_remaining']
    assert sorted(tiny.pairs()) == sorted(full.pairs())
    assert len(full) > 0


def test_annotation_budget_is_fail_closed(tmp_path):
    a = np.random.default_rng(17).integers(0, 8, (12, 12, 12), dtype=np.uint8)
    morse = CompactMorse(a, tmp_path/'catalog', indexed_scheduler=True)
    with pytest.raises(RuntimeError, match='annotation budget'):
        morse.h1_records(max_annotation_entries=1)
    assert not (morse.path/'h1_pairs_metadata.json').exists()
    with pytest.raises(ValueError, match='Positive cohomology'):
        morse.h1_records(max_annotation_entries=0)
    rows = morse.h1_records(max_annotation_entries=1_000_000)
    meta = json.loads((morse.path/'h1_pairs_metadata.json').read_text())
    assert meta['peak_annotation_entries'] > 1
    assert len(rows) == meta['count']
    assert (morse.path/'h1_pairs.bin').stat().st_size == 16*len(rows)
    assert len(rows) > 1


def test_barcode_disk_budget_is_fail_closed(tmp_path):
    a = np.random.default_rng(23).integers(0, 8, (12, 12, 12), dtype=np.uint8)
    morse = CompactMorse(a, tmp_path/'catalog', indexed_scheduler=True)
    with pytest.raises(RuntimeError, match='barcode output budget'):
        morse.h1_records(max_output_pairs=1)
    assert not (morse.path/'h1_pairs_metadata.json').exists()
    rows = morse.h1_records(max_output_pairs=1_000_000)
    assert len(rows) > 1


def test_two_identical_intervals_keep_distinct_certificates(tmp_path):
    v = np.full((12,12,12), 5, dtype=np.uint8)
    for start in (1,7):
        v[start:start+4,start:start+4,start:start+4]=0
        v[start:start+4,start+1,start+1]=5
    r=h1_persistence(v,output_dir=tmp_path,indexed_scheduler=True)
    assert r.pairs()==[(0,5),(0,5)]
    assert len(set(r.pair_ids))==2
    certs=[r.reconstruct(x) for x in r.pair_ids]
    assert all(x['verified'] for x in certs)
    assert certs[0]['cycle_cells']!=certs[1]['cycle_cells']


def test_independent_closed_voxel_betti_1(tmp_path):
    scipy = pytest.importorskip('scipy.ndimage')
    skimage = pytest.importorskip('skimage.measure')
    a = np.random.default_rng(11).integers(0, 8, (24, 24, 24), dtype=np.uint8)
    r=h1_persistence(a,output_dir=tmp_path,indexed_scheduler=True,
                     max_heap_entries=200,store_pair_ids=False)
    b1=[]
    for t in range(8):
        fg=a<=t
        _,c0=scipy.label(fg,structure=np.ones((3,3,3),dtype=np.uint8))
        labs,nb=scipy.label(~fg,structure=scipy.generate_binary_structure(3,1))
        faces=[labs[0],labs[-1],labs[:,0],labs[:,-1],labs[:,:,0],labs[:,:,-1]]
        bd=np.unique(np.concatenate([x.ravel() for x in faces]))
        c2=nb-len(bd)+int(0 in bd)
        chi=skimage.euler_number(fg,connectivity=3)
        b1.append(int(c0+c2-chi))
    assert np.array_equal(r.at(np.arange(8)),np.asarray(b1))
    assert b1[-1]==0
