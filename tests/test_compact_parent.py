"""Exactness and output-buffer tests for the compact streaming parent state."""
from __future__ import annotations
import numpy as np
import pytest
from betti_curves_py import persistence, load_hierarchical_pairs
from betti_curves_py.slab import slab_persistence

@pytest.mark.parametrize('dim', (0, 2))
@pytest.mark.parametrize('conn', (6, 26))
@pytest.mark.parametrize('dtype', (np.uint8, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize('chunk,disk', ((1, False), (7, True), (128, False)))
def test_compact_equals_global(tmp_path, dim, conn, dtype, chunk, disk):
    rng = np.random.default_rng(23 + dim + conn + chunk + np.dtype(dtype).itemsize)
    v = rng.integers(0, 13, (8, 7, 6)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        v = v / 4. - 1.5
    oracle = persistence(v, dim, conn, backend='numba-reference')
    r = slab_persistence(v, dim, tmp_path/'results', foreground_connectivity=conn,
                         slab_depth=3, tile_edge=4, write_curve=False,
                         intra_slab='boundary-aware', merge_strategy='compact-parent',
                         interval_chunk=chunk, disk_retained=disk)
    got = sorted(map(tuple, load_hierarchical_pairs(r).tolist()))
    expected = sorted(zip(oracle.births[:oracle.finite_count], oracle.deaths[:oracle.finite_count]))
    assert got == expected
    assert r.pair_count == oracle.finite_count
    assert r.essential_birth == (oracle.births[-1] if dim == 0 else None)

@pytest.mark.parametrize('dim,conn', ((0,6),(0,26),(2,6),(2,26)))
@pytest.mark.parametrize('disk', (False, True))
def test_compact_continuous_and_memory_mapped(tmp_path, dim, conn, disk):
    rng=np.random.default_rng(800 + dim + conn)
    arr=rng.uniform(-5, 4, size=(13,15,11)).astype(np.float32)
    source=tmp_path/'volume.npy'
    np.save(source, arr)
    oracle=persistence(arr, dim, conn, backend='numba-reference')
    r=slab_persistence(np.load(source,mmap_mode='r'), dim, tmp_path/'res',
                       foreground_connectivity=conn, slab_depth=4, tile_edge=6,
                       intra_slab='boundary-aware', merge_strategy='compact-parent',
                       event_page_mib=0 if disk else 256, interval_chunk=11,
                       disk_retained=disk, write_curve=False)
    assert sorted(map(tuple,load_hierarchical_pairs(r).tolist())) == sorted(
        zip(oracle.births[:oracle.finite_count],oracle.deaths[:oracle.finite_count]))

@pytest.mark.parametrize('dim', (0,2))
def test_compact_virtual_ids_above_32bit(tmp_path, dim):
    from betti_curves_py.hierarchical import _Graph, _Sink
    from betti_curves_py.parent_stream import stream_parent_reduce
    from betti_curves_py.certificate import CertificateWorkspace
    shape=(1_100_000_010,2,2)
    mid=1_100_000_005
    outside=int(np.prod(shape,dtype=np.int64))
    def graph(z):
        ids=np.arange(z*4,(z+1)*4,dtype=np.int64)
        values=np.array([1,2,3,4],np.uint16)
        edges=np.array([0,1,2],np.int32)
        if dim == 2:
            ids=np.r_[ids, outside]
            values=np.r_[values, np.uint16(0)]
            edges=np.r_[edges, 3].astype(np.int32)
            v=np.array([1,2,3,4],np.int32)
            levels=np.array([2,3,4,4],np.uint16)
        else:
            v=np.array([1,2,3],np.int32)
            levels=np.array([2,3,4],np.uint16)
        return _Graph(ids,values,edges,v,levels)
    sink=_Sink(tmp_path/'pairs.bin',np.uint16,curve=False)
    try:
        workspace=CertificateWorkspace(tmp_path/'temp',16)
        g,_,stats=stream_parent_reduce(graph(mid-1),graph(mid),mid+1,shape,6,
                                       dim==2,sink,False,workspace,16,
                                       compact_state=True,interval_chunk=1,disk_retained=True)
        assert stats['nodes']>=8
        assert np.all((g.ids[:-1] > 2**32))
    finally:
        sink.close()


def test_compact_cli(tmp_path):
    from betti_curves_py.cli import main
    p=tmp_path/'input.npy'
    np.save(p,np.random.default_rng(22).integers(0,10,(9,7,8),dtype=np.uint16))
    assert main([str(p),'--backend','slab','--mode','both',
                 '--slab-seam-strategy','compact-parent','--interval-chunk','3',
                 '--disk-retained','--tile-edge','4','--slab-depth','3',
                 '--intra-slab','boundary-aware','--output-dir',str(tmp_path/'out')]) == 0
    assert (tmp_path/'out'/'h2_pairs.bin').is_file()


def test_chunk_validation(tmp_path):
    with pytest.raises(ValueError,match='interval_chunk'):
        slab_persistence(np.zeros((2,3,3),np.uint8),0,tmp_path,
                         merge_strategy='compact-parent',interval_chunk=0)

@pytest.mark.parametrize('dim,conn', ((0,6),(0,26),(2,6),(2,26)))
@pytest.mark.parametrize('intra', ('full','two-face','boundary-forest'))
def test_compact_other_intra_slab_modes(tmp_path,dim,conn,intra):
    rng=np.random.default_rng(199+dim*5+conn)
    a=rng.integers(0,8,(9,9,8),dtype=np.uint16)
    expected=persistence(a,dim,conn,backend='numba-reference')
    r=slab_persistence(a,dim,tmp_path,foreground_connectivity=conn,
        slab_depth=3,tile_edge=4,merge_strategy='compact-parent',
        intra_slab=intra,interval_chunk=3,write_curve=False)
    assert sorted(map(tuple,load_hierarchical_pairs(r).tolist())) == sorted(
        zip(expected.births[:expected.finite_count],expected.deaths[:expected.finite_count]))
