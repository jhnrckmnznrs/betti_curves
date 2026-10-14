"""Construct compact original-boundary CSR without Python cell objects."""
from __future__ import annotations
import numpy as np
from numba import njit
from ._blockwise import blockwise_collapse
from ._levels import _full_levels

@njit(cache=True)
def _lookup(sorted_ids, key):
    lo=0;hi=len(sorted_ids)
    while lo<hi:
        mid=(lo+hi)//2
        if sorted_ids[mid]<key:lo=mid+1
        else:hi=mid
    if lo==len(sorted_ids) or sorted_ids[lo]!=key:return -1
    return lo

@njit(cache=True)
def _make_csr(ids, shape):
    nx,ny,nz=shape
    sy=2*ny+1;sz=2*nz+1
    strides=(sy*sz,sz,1)
    n=len(ids)
    dims=np.empty(n,dtype=np.uint8)
    offsets=np.zeros(n+1,dtype=np.int64)
    for i in range(n):
        p=int(ids[i]);coords=(p//strides[0], (p//sz)%sy, p%sz)
        d=0
        for a in range(3):
            if coords[a]&1:d+=1
        dims[i]=d;offsets[i+1]=offsets[i]+2*d
    data=np.empty(int(offsets[-1]),dtype=np.uint32)
    for i in range(n):
        p=int(ids[i]);coords=(p//strides[0], (p//sz)%sy, p%sz)
        k=int(offsets[i])
        for a in range(3):
            if coords[a]&1:
                left=_lookup(ids,p-strides[a]);right=_lookup(ids,p+strides[a])
                if left<0 or right<0:raise ValueError('Precollapse is not a subcomplex')
                data[k]=left;data[k+1]=right;k+=2
        # <=6 entries, sort for efficient native symmetric difference
        for s in range(int(offsets[i]),int(offsets[i+1])):
            for t in range(s+1,int(offsets[i+1])):
                if data[t]<data[s]:data[s],data[t]=data[t],data[s]
    return dims,offsets.astype(np.uint64),data

def prepare_complex(volume,*,pre_collapse=True,block_voxels=8,max_cells=3_000_000):
    v=np.asarray(volume)
    if v.ndim!=3 or v.dtype not in (np.uint8,np.uint16) or any(x<1 for x in v.shape):
        raise ValueError('Expected nonempty uint8 or uint16 volume')
    cells=int(np.prod([2*int(x)+1 for x in v.shape]))
    if cells>2**32-1 or cells>max_cells:raise MemoryError('Full cubical grid exceeds max_cells')
    if pre_collapse:
        c=blockwise_collapse(v,block_voxels=block_voxels,max_cells=max_cells)
        ids=np.ascontiguousarray(np.flatnonzero(c['active']),dtype=np.uint32)
        levels=c['levels'][ids].astype(np.int32,copy=False)
        stats={'full_cells':cells,'precollapse_remaining':len(ids),'precollapse_pairs':c['local_pairs']+c['global_pairs']}
    else:
        ids=np.arange(cells,dtype=np.uint32)
        lv=_full_levels(v)
        levels=lv.astype(np.int32,copy=False)
        stats={'full_cells':cells,'precollapse_remaining':len(ids),'precollapse_pairs':0}
    dims,offsets,data=_make_csr(ids,tuple(int(x) for x in v.shape))
    return {'ids':ids,'levels':np.ascontiguousarray(levels),'dims':dims,
            'offsets':offsets,'data':data,'shape':tuple(int(x) for x in v.shape),'stats':stats}
