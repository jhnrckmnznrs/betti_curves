"""Exactness and bounded-allocation checks for the streaming 26-Z seam."""
from __future__ import annotations

import numpy as np
import pytest
from betti_curves_py import persistence, load_hierarchical_pairs
from betti_curves_py.core import stable_voxel_order
from betti_curves_py.seam26 import generate_seam26_batch
from betti_curves_py.slab import slab_persistence


def _edges_from_generator(a, b, width, descending, batch_size):
    ao = stable_voxel_order(a, descending).astype(np.int32)
    bo = stable_voxel_order(b, descending).astype(np.int32)
    eu = np.empty(9*batch_size, dtype=np.int32)
    ev = np.empty_like(eu)
    ew = np.empty(9*batch_size, dtype=a.dtype)
    pl = pr = 0
    records = []
    while pl < len(a) or pr < len(a):
        pl2, pr2, count = generate_seam26_batch(a, b, ao, bo, 0, 0, len(a),
                                                  width, descending, pl, pr,
                                                  batch_size, eu, ev, ew)
        assert (pl2, pr2) != (pl, pr)
        pl, pr = pl2, pr2
        records.extend((int(eu[i]), int(ev[i])-len(a), float(ew[i])) for i in range(count))
    return records


@pytest.mark.parametrize('descending', (False, True))
@pytest.mark.parametrize('shape', ((2,3), (5,7), (9,8)))
@pytest.mark.parametrize('batch', (1, 3, 17))
def test_generator_exact_edges_and_filtration(descending, shape, batch):
    rng = np.random.default_rng(7+shape[0]+batch)
    a = rng.integers(0, 7, size=shape, dtype=np.uint16).ravel()
    b = rng.integers(0, 7, size=shape, dtype=np.uint16).ravel()
    result = _edges_from_generator(a, b, shape[1], descending, batch)
    expected = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            j = y*shape[1]+x
            for ny in range(max(0,y-1),min(shape[0],y+2)):
                for nx in range(max(0,x-1),min(shape[1],x+2)):
                    k = ny*shape[1]+nx
                    expected.append((j,k,float(min(a[j], b[k]) if descending else max(a[j],b[k]))))
    assert set(result) == set(expected)
    assert len(result) == len(expected)  # no duplicated edge
    levels = [e[2] for e in result]
    assert levels == sorted(levels, reverse=descending)


@pytest.mark.parametrize('dim,conn', ((0,26),(2,6),(0,6),(2,26)))
@pytest.mark.parametrize('dtype', (np.uint8,np.uint16,np.float32,np.float64))
@pytest.mark.parametrize('uf_layout', ('separate','interleaved'))
@pytest.mark.parametrize('batch', (1, 17))
def test_exact_pairs_across_layouts(tmp_path, dim, conn, dtype, uf_layout, batch):
    rng = np.random.default_rng(423 + dim*7 + conn + batch)
    v = rng.integers(0,11,(8,9,10)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        v = v / 4 - 1.5
    oracle = persistence(v, dim, conn, backend='numba-reference')
    r = slab_persistence(v,dim,tmp_path/'out',foreground_connectivity=conn,
                         slab_depth=3,tile_edge=5,write_curve=False,
                         merge_strategy='stream26-blocked',intra_slab='boundary-aware',
                         interval_chunk=3,seam_batch_activations=batch,
                         uf_layout=uf_layout,uf_block_entries=17)
    got = sorted(map(tuple,load_hierarchical_pairs(r).tolist()))
    expected = sorted(zip(oracle.births[:oracle.finite_count],oracle.deaths[:oracle.finite_count]))
    assert got == expected
    assert r.essential_birth == (oracle.births[-1] if dim==0 else None)


@pytest.mark.parametrize('dim,conn', ((0,26),(2,6)))
@pytest.mark.parametrize('intra', ('full','two-face','boundary-forest'))
@pytest.mark.parametrize('mapped', (False,True))
def test_modes_and_memmap(tmp_path, dim,conn,intra,mapped):
    rng = np.random.default_rng(932+dim+conn)
    a = rng.uniform(-2,7,(11,13,12)).astype(np.float32)
    path = tmp_path/'input.npy'
    np.save(path,a)
    v=np.load(path,mmap_mode='r') if mapped else a
    oracle=persistence(a,dim,conn,backend='numba-reference')
    result=slab_persistence(v,dim,tmp_path/'res',foreground_connectivity=conn,
                             slab_depth=4,tile_edge=6,merge_strategy='stream26-blocked',
                             intra_slab=intra,interval_chunk=5,write_curve=False,
                             event_page_mib=0.0 if mapped else 128.,
                             disk_retained=mapped,seam_batch_activations=3)
    expected=sorted(zip(oracle.births[:oracle.finite_count],oracle.deaths[:oracle.finite_count]))
    assert sorted(map(tuple,load_hierarchical_pairs(result).tolist())) == expected


def test_cli(tmp_path):
    from betti_curves_py.cli import main
    src=tmp_path/'v.npy'
    np.save(src,np.random.default_rng(22).integers(0, 7,(6,7,8),dtype=np.uint16))
    assert main([str(src),'--backend','slab','--mode','both',
                 '--slab-seam-strategy','stream26-blocked',
                 '--uf-layout','interleaved', '--seam-batch-activations','3',
                 '--uf-block-entries','64', '--interval-chunk','7',
                 '--tile-edge','4','--slab-depth','2',
                 '--intra-slab','boundary-aware','--output-dir',str(tmp_path/'out')]) == 0
    assert (tmp_path/'out'/'h2_pairs.bin').is_file()


def test_validation(tmp_path):
    a=np.zeros((3,3,3),np.uint8)
    for name, val in [('seam_batch_activations',0), ('uf_block_entries',0),('uf_layout','invalid')]:
        with pytest.raises(ValueError):
            slab_persistence(a,0,tmp_path,merge_strategy='stream26-blocked', **{name:val})

@pytest.mark.parametrize('dim', (0,2))
def test_stream26_virtual_id_over_uint32(tmp_path, dim):
    """Local 32-bit roots must not truncate global 64-bit witness IDs."""
    from betti_curves_py.hierarchical import _Graph, _Sink
    from betti_curves_py.parent_stream import stream_parent_reduce
    from betti_curves_py.certificate import CertificateWorkspace
    shape=(1_100_000_010, 2, 2)
    mid=1_100_000_005
    outside=int(np.prod(shape,dtype=np.int64))
    def graph(z):
        ids=np.arange(z*4,(z+1)*4,dtype=np.int64)
        values=np.array([1,2,3,4],np.uint16)
        u=np.array([0,1,2],np.int32)
        v=np.array([1,2,3],np.int32)
        levels=np.array([2,3,4],np.uint16)
        if dim==2:
            ids=np.r_[ids,outside]
            values=np.r_[values,np.uint16(0)]
            u=np.r_[u,np.int32(3)]
            v=np.r_[v,np.int32(4)]
            levels=np.r_[levels,np.uint16(4)]
        return _Graph(ids,values,u,v,levels)
    sink = _Sink(tmp_path/'pairs.bin',np.uint16,curve=False)
    try:
        workspace=CertificateWorkspace(tmp_path/'tmp',16)
        g,_,stats=stream_parent_reduce(graph(mid-1),graph(mid),mid+1,shape,26,
                    dim==2,sink,False,workspace,16,compact_state=True,
                    stream_26=True,seam_batch_activations=1,interval_chunk=1,
                    uf_layout='interleaved')
        assert stats['seam_events'] > 0
        assert g.ids[0] > 2**32
    finally:
        sink.close()

@pytest.mark.parametrize('dim,conn', ((0,26),(2,6)))
def test_medium_random_against_oracle(tmp_path, dim,conn):
    rng=np.random.default_rng(2012+dim)
    arr=rng.integers(0,43,(20,40,48),dtype=np.uint16)
    expected=persistence(arr,dim,conn,backend='numba-reference')
    r=slab_persistence(arr,dim,tmp_path/'result',foreground_connectivity=conn,
                       slab_depth=5,tile_edge=12,merge_strategy='stream26-blocked',
                       intra_slab='boundary-aware',interval_chunk=256,
                       seam_batch_activations=12,write_curve=False)
    assert sorted(map(tuple,load_hierarchical_pairs(r).tolist())) == sorted(
        zip(expected.births[:expected.finite_count],expected.deaths[:expected.finite_count]))
