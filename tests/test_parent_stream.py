"""Full barcode exactness tests for the fused, never-materialized Z parent."""
from __future__ import annotations

import numpy as np
import pytest

from betti_curves_py import persistence, load_hierarchical_pairs
from betti_curves_py.slab import slab_persistence


@pytest.mark.parametrize('dimension', (0, 2))
@pytest.mark.parametrize('foreground_connectivity', (6, 26))
@pytest.mark.parametrize('dtype', (np.uint8, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize('intra_slab', ('full', 'boundary-aware'))
@pytest.mark.parametrize('slab_depth', (2, 4))
def test_stream_parent_matches_global(tmp_path, dimension, foreground_connectivity,
                                       dtype, intra_slab, slab_depth):
    seed = (dimension*111 + foreground_connectivity*23 + slab_depth*19 +
            int(np.dtype(dtype).itemsize)*3)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 13, size=(7, 6, 7)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        arr = arr / 4.0 - 0.75
    oracle = persistence(arr, dimension, foreground_connectivity, backend='numba-reference')
    result = slab_persistence(arr, dimension, tmp_path,
        foreground_connectivity=foreground_connectivity,
        slab_depth=slab_depth, tile_edge=3, write_curve=False,
        intra_slab=intra_slab, merge_strategy='stream-parent')
    got = sorted(map(tuple, load_hierarchical_pairs(result).tolist()))
    expected = sorted(zip(oracle.births[:oracle.finite_count], oracle.deaths[:oracle.finite_count]))
    assert got == expected
    assert result.pair_count == oracle.finite_count
    if dimension == 0:
        assert result.essential_birth == oracle.births[-1]
    else:
        assert result.essential_birth is None


@pytest.mark.parametrize('dim,conn', ((0,6),(0,26),(2,6),(2,26)))
@pytest.mark.parametrize('dtype', (np.uint16, np.float32))
def test_continuous_memmap_and_forced_paging(tmp_path, dim, conn, dtype):
    rng = np.random.default_rng(1090+dim+conn)
    volume = rng.uniform(-2.5, 7.0, size=(10, 12, 11)).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        volume = (np.abs(volume)*100).astype(dtype)
    path = tmp_path / 'vol.npy'
    np.save(path, volume)
    oracle = persistence(volume, dim, conn, backend='numba-reference')
    r = slab_persistence(np.load(path, mmap_mode='r'), dim, tmp_path/'res',
        foreground_connectivity=conn, slab_depth=3, tile_edge=4,
        intra_slab='boundary-aware', merge_strategy='stream-parent',
        event_page_mib=0, write_curve=False)
    assert sorted(map(tuple, load_hierarchical_pairs(r).tolist())) == sorted(
        zip(oracle.births[:oracle.finite_count], oracle.deaths[:oracle.finite_count]))


def test_stream_parent_cli(tmp_path):
    from betti_curves_py.cli import main
    rng = np.random.default_rng(19)
    p = tmp_path / 'v.npy'
    np.save(p, rng.integers(0,5,(7,6,5),dtype=np.uint16))
    assert main([str(p), '--backend','slab', '--mode','both',
                 '--slab-seam-strategy','stream-parent',
                 '--slab-depth','2','--tile-edge','3',
                 '--intra-slab','boundary-aware',
                 '--output-dir',str(tmp_path/'output')]) == 0
    assert (tmp_path/'output'/'h0_pairs.bin').exists()
    assert (tmp_path/'output'/'h2_pairs.bin').exists()

@pytest.mark.parametrize('dimension,foreground_connectivity', ((0,6),(0,26),(2,6),(2,26)))
def test_medium_random_barcode(tmp_path, dimension, foreground_connectivity):
    rng=np.random.default_rng(920+dimension+foreground_connectivity)
    vol=rng.integers(0,256,(24,28,30),dtype=np.uint16)
    oracle=persistence(vol,dimension,foreground_connectivity,backend='numba-reference')
    res=slab_persistence(vol, dimension, tmp_path, foreground_connectivity=foreground_connectivity,
                         slab_depth=5, tile_edge=12, intra_slab='boundary-aware',
                         merge_strategy='stream-parent',write_curve=False)
    assert sorted(map(tuple,load_hierarchical_pairs(res).tolist()))==sorted(
        zip(oracle.births[:oracle.finite_count],oracle.deaths[:oracle.finite_count]))

@pytest.mark.parametrize('dimension', (0,2))
def test_virtual_global_ids_above_uint32(tmp_path,dimension):
    from betti_curves_py.hierarchical import _Graph, _Sink
    from betti_curves_py.certificate import CertificateWorkspace
    from betti_curves_py.parent_stream import stream_parent_reduce
    # The virtual volume is huge, but each test child has only four voxels.
    shape=(1_100_000_010,2,2)
    mid=1_100_000_005
    p=4
    outside=np.prod(shape,dtype=np.int64)
    def graph(z):
        ids=np.arange(z*p,(z+1)*p,dtype=np.int64)
        values=np.array([1,2,3,4],np.uint16)
        edges=np.array([0,1,2],np.int32)
        if dimension==2:
            ids=np.concatenate((ids,np.array([outside],np.int64)))
            values=np.concatenate((values,np.array([0],np.uint16)))
            edges=np.concatenate((edges,np.array([3],np.int32)))
            v=np.array([1,2,3,4],np.int32)
            levels=np.array([2,3,4,4],np.uint16)
        else:
            v=np.array([1,2,3],np.int32)
            levels=np.array([2,3,4],np.uint16)
        return _Graph(ids,values,edges,v,levels)
    sink=_Sink(tmp_path/'pairs.bin',np.uint16,curve=False)
    try:
        space=CertificateWorkspace(tmp_path/'scratch',64)
        g,essential,stats=stream_parent_reduce(graph(mid-1),graph(mid),mid+1,
            shape,6,dimension==2,sink,False,space,64)
        assert all(int(i)>2**32 for i in g.ids[:-1] if int(i)!=outside)
        assert stats['nodes']>=8
    finally:
        sink.close()
