"""H1 persistence integration and independent cubical oracle checks (CPU)."""
import numpy as np
import pytest
from betti_curves_py.h1_morse import (
    H1MorseResult, analyze_volume, h1_persistence,
)
from betti_curves_py._morse._csr import prepare_complex


def oracle_h1(volume):
    """Independent full-boundary F2 reduction, practical only for tiny grids."""
    prep = prepare_complex(volume, pre_collapse=False, max_cells=10000)
    ids, dims, levels, ptr, data = (prep[k] for k in ('ids','dims','levels','offsets','data'))
    order = sorted(range(len(ids)), key=lambda i:(int(levels[i]),int(dims[i]),int(ids[i])))
    rank={idx:j for j,idx in enumerate(order)}
    reduced={}; found=[]
    for c in order:
        col=set(map(int,data[int(ptr[c]):int(ptr[c+1])]))
        while col:
            p=max(col,key=rank.__getitem__)
            if p not in reduced:break
            col.symmetric_difference_update(reduced[p])
        if col:
            p=max(col,key=rank.__getitem__)
            reduced[p]=col
            if dims[p]==1 and levels[p]<levels[c]:
                found.append((int(levels[p]),int(levels[c])))
    return sorted(found)


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_h1_matches_full_cubical_reduction(tmp_path,seed,dtype):
    volume=np.random.default_rng(18+seed).integers(0,6,(3,3,3),dtype=dtype)
    r=h1_persistence(volume,output_dir=tmp_path / f'run{seed}',max_cells=10000)
    assert sorted(r.pairs())==oracle_h1(volume)
    assert r.at(np.arange(6)).tolist()==[
        sum(b <= t < d for b,d in r.pairs()) for t in range(6)
    ]
    assert len(set(r.pair_ids))==len(r)
    for pid in r.pair_ids[:2]:
        cert=r.reconstruct(pid)
        assert cert['verified'] and cert['pair_id']==pid


def test_closed_voxel_convention_is_required(tmp_path):
    v=np.zeros((3,3,3),dtype=np.uint8)
    with pytest.raises(ValueError,match='26-connected'):
        h1_persistence(v,connectivity=6,output_dir=tmp_path)
    with pytest.raises(ValueError,match='26-connected'):
        analyze_volume(v,connectivity=6,h1_backend='morse',output_dir=tmp_path)
    with pytest.raises(TypeError,match='uint8 and uint16'):
        h1_persistence(v.astype('float32'),output_dir=tmp_path)
    with pytest.raises(ValueError,match='3D'):
        h1_persistence(v[0],output_dir=tmp_path)


def test_zero_intervals_and_budget(tmp_path):
    v=np.zeros((3,3,3),dtype=np.uint8)
    r=h1_persistence(v,output_dir=tmp_path)
    assert not len(r) and r.betti_curve()==[]
    assert r.at(0)==0 and r.diagram_rows().shape==(0,2)
    with pytest.raises(MemoryError,match='max_cells'):
        h1_persistence(v,output_dir=tmp_path/'small',max_cells=10)
    with pytest.raises(ValueError,match='not found'):
        r.reconstruct('none')


def test_duplicate_birth_death_preserves_pair_ids(tmp_path):
    v=np.full((12,12,12),5,dtype=np.uint8)
    v[1:5,1:5,1:5]=0
    v[7:11,7:11,7:11]=0
    v[1:5,2,2]=5
    v[7:11,8,8]=5
    r=h1_persistence(v,output_dir=tmp_path)
    assert sorted(r.pairs())==[(0,5),(0,5)]
    assert len(set(r.pair_ids))==2
    c=[r.reconstruct(pid) for pid in r.pair_ids]
    assert all(x['verified'] for x in c)
    assert c[0]['cycle_cells'] != c[1]['cycle_cells']


def test_array_and_mmap_input(tmp_path):
    v=np.full((10,10,10),5,dtype=np.uint16)
    v[1:9,1:9,1:9]=0
    v[1:9,5,5]=5
    np.save(tmp_path/'vol.npy',v)
    r=h1_persistence(tmp_path/'vol.npy',output_dir=tmp_path/'catalog')
    assert r.pairs()==[(0,5)]
    reopened=r.reconstruct(r.pair_ids[0])
    assert reopened['verified']


def test_scratch_lifecycle():
    v=np.zeros((2,2,2),dtype=np.uint8)
    r=h1_persistence(v)
    p=r.catalog_path
    assert p.exists()
    r.close()
    assert not p.exists()


def test_result_betti_curve_half_open():
    r=H1MorseResult(
        np.array([0,1],dtype=np.uint8),
        np.array([2,3],dtype=np.uint8),
        ('a','b'),catalog_path=None,
    )
    assert r.betti_curve()==[(0,1),(1,2),(2,1),(3,0)]
    assert r.at(np.array([-1,0,1,2,3,4])).tolist()==[0,1,2,1,0,0]


def test_betti_zero_shortcut_is_exact(tmp_path):
    """Only runs when host betti_curves H0/H2/Euler modules are installed."""
    pytest.importorskip('betti_curves_py.core')
    pytest.importorskip('betti_curves_py.euler_stream')
    v=np.zeros((5,5,5),dtype=np.uint8)
    r=analyze_volume(v,connectivity=26,h1_backend='morse',output_dir=tmp_path)
    assert r.validated and r.h1.skipped_empty
    assert len(r.h1)==0 and r.h1.catalog_path is None


@pytest.mark.parametrize('seed',[1,5,12])
def test_independent_h1_betti_curve(tmp_path,seed):
    pytest.importorskip('betti_curves_py.core')
    pytest.importorskip('betti_curves_py.euler_stream')
    v=np.random.default_rng(seed).integers(0,6,(4,4,4),dtype=np.uint8)
    r=analyze_volume(v,connectivity=26,h1_backend='morse',
                     output_dir=tmp_path/'catalog', max_cells=10000)
    assert r.validated
    ts=np.arange(6)
    assert r.h1.at(ts).tolist()==[
        sum(b<=t<d for b,d in r.h1.pairs()) for t in ts
    ]
    assert r.h1_betti_curve==r.h1.betti_curve()


def test_curve_only_works_with_6_connected(tmp_path):
    pytest.importorskip('betti_curves_py.core')
    pytest.importorskip('betti_curves_py.euler_stream')
    v=np.zeros((3,3,3),dtype=np.uint8)
    r=analyze_volume(v,connectivity=6,h1_backend='curve')
    assert r.h1 is None and r.h1_betti_curve==[]
