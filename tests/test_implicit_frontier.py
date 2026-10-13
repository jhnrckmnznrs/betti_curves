"""Exactness and storage invariants for v0.10 implicit Z-frontier."""
import json
import numpy as np
import pytest
from betti_curves_py import persistence, slab_persistence, load_hierarchical_pairs
from betti_curves_py.hierarchical import _Graph
from betti_curves_py.implicit_frontier import pack_frontier


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32,np.float64])
@pytest.mark.parametrize('dim',[0,2])
@pytest.mark.parametrize('conn',[6,26])
@pytest.mark.parametrize('intra',['full','boundary-aware','two-face'])
@pytest.mark.parametrize('page',[0.,128.])
def test_exact_random(tmp_path,dtype,dim,conn,intra,page):
    seed=100+int(dim)*37+conn+(0 if intra=='full' else 7)
    rng=np.random.default_rng(seed)
    arr=rng.integers(0,25,size=(8,10,11)).astype(dtype)
    if np.dtype(dtype).kind=='f':
        arr=(arr / dtype(13)) - dtype(1.5)
    expected=persistence(arr,dim,conn,backend='numba-reference')
    result=slab_persistence(arr,dim,tmp_path,foreground_connectivity=conn,
              slab_depth=3,tile_edge=4,write_curve=False,merge_strategy='paged-tree',
              intra_slab=intra,event_page_mib=page,implicit_frontier=True)
    got=sorted(map(tuple,load_hierarchical_pairs(result).tolist()))
    ref=sorted(zip(expected.births[:expected.finite_count],expected.deaths[:expected.finite_count]))
    assert got==ref
    assert result.essential_birth==(expected.births[-1] if dim==0 else None)
    assert json.loads(result.metadata_path.read_text())['implicit_frontier'] is True


@pytest.mark.parametrize('dim',[0,2])
@pytest.mark.parametrize('conn',[6,26])
def test_continuous_mmap(tmp_path,dim,conn):
    rng=np.random.default_rng(4331+dim+conn)
    a=rng.uniform(-4,7,size=(13,12,15)).astype(np.float32)
    np.save(tmp_path/'a.npy',a)
    mapped=np.load(tmp_path/'a.npy',mmap_mode='r')
    expected=persistence(a,dim,conn,backend='numba-reference')
    r=slab_persistence(mapped,dim,tmp_path/'out',foreground_connectivity=conn,
                  slab_depth=4,tile_edge=5,write_curve=False,
                  intra_slab='boundary-aware',merge_strategy='paged-tree',implicit_frontier=True)
    assert sorted(map(tuple,load_hierarchical_pairs(r)))==sorted(zip(
        expected.births[:expected.finite_count],expected.deaths[:expected.finite_count]))


def test_implicit_storage_roundtrip_large_ids():
    # A tiny virtual face near ID 2**33: ID reconstruction cannot use uint32.
    base=2**33
    shape=(base//12+100,3,4)
    exterior=int(np.prod(shape,dtype=np.int64))
    ids=np.r_[np.array([base-99,base-17],np.int64),
              np.arange(base,base+12,dtype=np.int64),exterior]
    vals=np.arange(len(ids),dtype=np.uint16)
    u=np.array([0,2,4,13],np.int32)
    v=np.array([1,3,6,14],np.int32)
    g=_Graph(ids,vals,u,v,np.arange(len(u),dtype=np.uint16))
    # z_end is derived from the start of the contiguous face.
    # Choose a virtual physical plane index that matches 'base'.
    assert base%12!=0 or True
    face_start=(base//12)*12
    ids[2:14]=np.arange(face_start,face_start+12,dtype=np.int64)
    g=_Graph(ids,vals,u,v,g.levels)
    packed=pack_frontier(g,base//12+1,shape,True)
    restored=packed.materialize()
    for field in ('ids','values','u','v','levels'):
        assert np.array_equal(getattr(restored,field),getattr(g,field))
    assert packed.nbytes < sum(getattr(g,k).nbytes for k in ('ids','values','u','v','levels'))


def test_invalid_merge_setting(tmp_path):
    with pytest.raises(ValueError,match='requires'):
        slab_persistence(np.zeros((2,3,4),np.uint8),0,tmp_path,
                         merge_strategy='forest',implicit_frontier=True)
