"""Filtration-exact local elementary collapses, followed by optional global sweep.

A local pair is removed ONLY if the lower face is globally free, its unique
coface is globally maximal, both have equal filtration level, and both belong
to the same spatial tile. The remaining active cells form an actual subcomplex
of the original grid; therefore inclusion is the cycle-lifting map.

This is conservative blockwise Morse simplification, not general Forman matching
across the tile interface. Memory is O(cubical cells), NOT out-of-core.
"""
from __future__ import annotations
import time
import numpy as np
from numba import njit
from ._levels import _full_levels

@njit(cache=True)
def _pass(levels,active,nx,ny,nz,block_voxels,local_only,reverse_initial):
    sy,sz=2*ny+1,2*nz+1
    total=len(levels)
    strides=(sy*sz,sz,1)
    sizes=(2*nx+1,sy,sz)
    queued=np.zeros(total,dtype=np.uint8)
    counts=np.zeros(total,dtype=np.uint8)
    stack=np.empty(total,dtype=np.int32)
    for p in range(total):
        if active[p]==0: continue
        z=p%sz;y=(p//sz)%sy;x=p//(sy*sz)
        cs=(x,y,z)
        count=0
        for a in range(3):
            if cs[a]&1:continue
            if cs[a]>0 and active[p-strides[a]]: count+=1
            if cs[a]<sizes[a]-1 and active[p+strides[a]]:count+=1
        counts[p]=count
    top=0
    for step in range(total):
        p=total-1-step if reverse_initial else step
        if active[p] and counts[p]==1:
            stack[top]=p;top+=1;queued[p]=1
    paired=0
    dim_stats=np.zeros(3,dtype=np.int64)
    while top:
        top-=1;p=np.int64(stack[top]);queued[p]=0
        if not active[p] or counts[p]!=1:continue
        z=p%sz;y=(p//sz)%sy;x=p//(sy*sz)
        coords=(x,y,z)
        tau=-1
        for a in range(3):
            if coords[a]&1:continue
            for d in (-1,1):
                if coords[a]+d<0 or coords[a]+d>=sizes[a]:continue
                q=p+d*strides[a]
                if active[q]:tau=q
        if tau<0 or counts[tau]!=0 or levels[p]!=levels[tau]:continue
        if local_only:
            tz=tau%sz;ty=(tau//sz)%sy;tx=tau//(sy*sz)
            if ((x//(2*block_voxels)!=tx//(2*block_voxels)) or
                (y//(2*block_voxels)!=ty//(2*block_voxels)) or
                (z//(2*block_voxels)!=tz//(2*block_voxels))):continue
        dim=(x&1)+(y&1)+(z&1)
        active[p]=0;active[tau]=0;paired+=1;dim_stats[dim]+=1
        for removed in (p,tau):
            rz=removed%sz;ry=(removed//sz)%sy;rx=removed//(sy*sz)
            rc=(rx,ry,rz)
            for a in range(3):
                if not (rc[a]&1):continue
                for off in (-strides[a],strides[a]):
                    f=removed+off
                    if not active[f]:continue
                    counts[f]-=1
                    if not queued[f]:
                        stack[top]=f;queued[f]=1;top+=1
                    if counts[f]==0:
                        fz=f%sz;fy=(f//sz)%sy;fx=f//(sy*sz)
                        fc=(fx,fy,fz)
                        for b in range(3):
                            if not (fc[b]&1):continue
                            for off2 in (-strides[b],strides[b]):
                                g=f+off2
                                if active[g] and not queued[g]:
                                    stack[top]=g;queued[g]=1;top+=1
    return dim_stats,paired


def blockwise_collapse(volume,*,block_voxels=16,global_finish=True,
                       max_cells=12_000_000,reverse_initial=False):
    """Return full-grid active mask, levels, and measurable per-pass statistics.

    The finite filtration is voxel-minimum / lower-star, using uint8/uint16.
    This routine is in-memory, with an explicit maximum number of grid cells.
    """
    v=np.asarray(volume)
    if v.ndim!=3 or v.dtype not in (np.uint8,np.uint16) or any(s<=0 for s in v.shape):
        raise ValueError('Nonempty uint8/uint16 3D volume required')
    if not isinstance(block_voxels,int) or block_voxels<1:
        raise ValueError('block_voxels must be a positive integer')
    total=int(np.prod([2*int(s)+1 for s in v.shape]))
    if total>=2**31:raise MemoryError('Packed cell count exceeds int32 stack addressing')
    if total>max_cells:raise MemoryError(f'{total} grid cells exceed limit {max_cells}')
    t=time.perf_counter()
    levels=_full_levels(v)
    active=np.ones(total,dtype=np.uint8)
    local, nlocal=_pass(levels,active,*v.shape,block_voxels,True,reverse_initial)
    if global_finish:
        global_stats,nglobal=_pass(levels,active,*v.shape,block_voxels,False,reverse_initial)
    else:
        global_stats=np.zeros(3,dtype=np.int64);nglobal=0
    if not active.any():raise AssertionError('Invalid collapse: all cells removed')
    return {'active':active,'levels':levels,'total':total,'retained':int(active.sum()),
            'block_voxels':block_voxels,'local_pairs':int(nlocal),
            'global_pairs':int(nglobal),'local_by_dimension':local,
            'global_by_dimension':global_stats,'seconds':time.perf_counter()-t,
            'shape':tuple(v.shape)}
